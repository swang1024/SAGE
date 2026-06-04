import argparse
import os
import sys
import logging
import hashlib
import time
from datetime import datetime

# Force Python to use the local mem0 package instead of the globally installed pip version
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import METHODS, TECHNIQUES


def _configure_mem0_local_runtime(args):
    """Configure mem0 local runtime env before importing mem0 modules.

    Local mem0 creates an internal telemetry Qdrant store under MEM0_DIR
    (e.g. ~/.mem0/migrations_qdrant). That shared default path causes file-lock
    collisions when multiple local runs start in parallel.
    """
    uses_local_mem0 = args.technique_type == "sage" or (
        args.technique_type == "mem0" and args.mem0_backend == "ollama"
    )
    if not uses_local_mem0:
        return

    # Default to disabling telemetry for local ablation runs to avoid creating
    # an extra internal Qdrant instance that can lock shared paths.
    if os.environ.get("MEM0_TELEMETRY") is None:
        os.environ["MEM0_TELEMETRY"] = "False"

    # If the caller did not set MEM0_DIR, isolate mem0 runtime files per process.
    if os.environ.get("MEM0_DIR") is None:
        qdrant_abs = os.path.abspath(args.qdrant_path)
        qdrant_hash = hashlib.sha1(qdrant_abs.encode("utf-8")).hexdigest()[:10]
        runtime_dir = os.path.join(
            "/tmp",
            f"mem0_runtime_{args.collection_name}_{qdrant_hash}_{os.getpid()}",
        )
        os.makedirs(runtime_dir, exist_ok=True)
        os.environ["MEM0_DIR"] = runtime_dir


class Experiment:
    def __init__(self, technique_type, chunk_size):
        self.technique_type = technique_type
        self.chunk_size = chunk_size

    def run(self):
        print(f"Running experiment with technique: {self.technique_type}, chunk size: {self.chunk_size}")


def _format_elapsed_time(seconds):
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{int(hours)}h {int(minutes)}m {sec:.2f}s"
    if minutes:
        return f"{int(minutes)}m {sec:.2f}s"
    return f"{sec:.2f}s"


def _log_phase_duration(phase_name, start_time):
    elapsed_seconds = time.perf_counter() - start_time
    elapsed_readable = _format_elapsed_time(elapsed_seconds)
    message = f"[TIMER] {phase_name} completed in {elapsed_readable}"
    print(message)
    logging.info(message)
    return elapsed_seconds


def main():
    script_started_at = time.perf_counter()
    script_started_wall = datetime.now()

    parser = argparse.ArgumentParser(description="Run the LOCOMO memory benchmark for a chosen technique/backend")
    parser.add_argument("--technique_type", choices=TECHNIQUES, default="mem0", help="Memory technique to use")
    parser.add_argument("--method", choices=METHODS, default="add", help="Method to use")
    parser.add_argument("--dataset_path", type=str, default="dataset/locomo10.json", help="Path to LOCOMO dataset")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for processing")
    parser.add_argument("--output_folder", type=str, default="results/", help="Output path for results")
    parser.add_argument("--top_k", type=int, default=30, help="Number of top memories to retrieve")
    parser.add_argument("--filter_memories", action="store_true", default=False, help="Whether to filter memories")
    parser.add_argument("--is_graph", action="store_true", default=False, help="Whether to use graph-based search")
    parser.add_argument("--num_chunks", type=int, default=1, help="Number of chunks to process")
    parser.add_argument(
        "--mem0_backend",
        choices=["cloud", "ollama", "openai"],
        default="cloud",
        help="Mem0 backend for ablation add runs",
    )
    parser.add_argument("--max_workers", type=int, default=10, help="Worker threads for add processing")
    parser.add_argument("--batch_size", type=int, default=2, help="Message batch size for add calls")
    parser.add_argument(
        "--infer_add",
        action="store_true",
        default=False,
        help="Enable inferred memory actions (ADD/UPDATE/DELETE) for local mem0 add runs",
    )
    parser.add_argument("--ollama_base_url", type=str, default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument("--llm_model", type=str, default="llama3.2", help="Ollama LLM model for local mem0")
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="nomic-embed-text",
        help="Ollama embedding model for local mem0",
    )
    parser.add_argument("--embedding_dims", type=int, default=768, help="Embedding dimensions")
    parser.add_argument(
        "--qdrant_path",
        type=str,
        default="/tmp/mem0_abla_qdrant_persist",
        help="Local qdrant path for mem0 ollama backend",
    )
    
    parser.add_argument(
        "--history_db_path",
        type=str,
        default="/tmp/mem0_abla_history_persist.db",
        help="History DB path for mem0 ollama backend",
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="mem0_abla_persist",
        help="Collection name for mem0 ollama backend",
    )
    parser.add_argument(
        "--action_stats_file",
        type=str,
        default="results/mem0_action_stats_abla.json",
        help="Path to write mem0 add action stats for ablation runs",
    )

    args = parser.parse_args()

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pid = os.getpid()

    # Append run stamp and PID to paths for consistency
    args.qdrant_path = f"{args.qdrant_path}_{run_stamp}_{pid}"
    if args.history_db_path.endswith('.db'):
        args.history_db_path = f"{args.history_db_path[:-3]}_{run_stamp}_{pid}.db"
    else:
        args.history_db_path = f"{args.history_db_path}_{run_stamp}_{pid}.db"
        
    if args.action_stats_file.endswith('.json'):
        args.action_stats_file = f"{args.action_stats_file[:-5]}_{run_stamp}_{pid}.json"
    else:
        args.action_stats_file = f"{args.action_stats_file}_{run_stamp}_{pid}.json"

    # Must run before importing modules that import mem0.memory.telemetry/setup.
    _configure_mem0_local_runtime(args)

    # Create a per-run log filename to avoid truncation/collisions across runs.
    os.makedirs("results", exist_ok=True)
    safe_model = args.llm_model.replace(":", "-").replace("/", "-")
    log_filename = (
        f"results/logs_{args.technique_type}_{args.method}_{safe_model}_vmf_{run_stamp}_{pid}.log"
    )

    logging.basicConfig(
        filename=log_filename,
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filemode="a",
        force=True,
    )
    print(f"Running experiments with technique: {args.technique_type}, chunk size: {args.chunk_size}")
    print(f"Log file: {log_filename}")

    try:
        if args.technique_type == "mem0":
            from src.memzero.add_abla import MemoryADDAbla

            if args.method in ("add", "full"):
                add_started_at = time.perf_counter()
                memory_manager = MemoryADDAbla(
                    data_path=args.dataset_path,
                    batch_size=args.batch_size,
                    is_graph=args.is_graph,
                    stats_output_path=args.action_stats_file,
                    backend=args.mem0_backend,
                    infer_add=args.infer_add,
                    ollama_base_url=args.ollama_base_url,
                    llm_model=args.llm_model,
                    embedding_model=args.embedding_model,
                    embedding_dims=args.embedding_dims,
                    qdrant_path=args.qdrant_path,
                    history_db_path=args.history_db_path,
                    collection_name=args.collection_name,
                )
                memory_manager.process_all_conversations(max_workers=args.max_workers)
                _log_phase_duration("mem0 add phase", add_started_at)

                # Release the Qdrant file lock so the search phase can open the same path
                if hasattr(memory_manager, "memory") and memory_manager.memory is not None:
                    if hasattr(memory_manager.memory, "vector_store"):
                        vs = memory_manager.memory.vector_store
                        if hasattr(vs, "client") and hasattr(vs.client, "close"):
                            vs.client.close()
                    if hasattr(memory_manager.memory, "db"):
                        memory_manager.memory.db.close()
                del memory_manager
            if args.method in ("search", "full"):
                search_started_at = time.perf_counter()
                if args.mem0_backend == "cloud":
                    from src.memzero.search import MemorySearch

                    output_file_path = os.path.join(
                        args.output_folder,
                        f"mem0_results_top_{args.top_k}_filter_{args.filter_memories}_graph_{args.is_graph}.json",
                    )
                    memory_searcher = MemorySearch(output_file_path, args.top_k, args.filter_memories, args.is_graph)
                    memory_searcher.process_data_file(args.dataset_path)
                elif args.mem0_backend == "ollama":
                    from src.memzero.search_local_backend import MemorySearchLocal

                    output_file_path = os.path.join(
                        args.output_folder,
                        f"mem0_{args.llm_model}_vmf_{run_stamp}_{os.getpid()}.json",
                    )
                    search_manager = MemorySearchLocal(
                        output_path=output_file_path,
                        top_k=args.top_k,
                        ollama_base_url=args.ollama_base_url,
                        llm_model=args.llm_model,
                        embedding_model=args.embedding_model,
                        embedding_dims=args.embedding_dims,
                        qdrant_path=args.qdrant_path,
                        history_db_path=args.history_db_path,
                        collection_name=args.collection_name,
                    )
                    search_manager.process_data_file(args.dataset_path)
                elif args.mem0_backend == "openai":
                    from src.memzero.search_local_backend import MemorySearchLocal

                    output_file_path = os.path.join(
                        args.output_folder,
                        f"mem0_{args.llm_model}_vmf_{run_stamp}_{os.getpid()}.json",
                    )
                    search_manager = MemorySearchLocal(
                        output_path=output_file_path,
                        top_k=args.top_k,
                        llm_model=args.llm_model,
                        embedding_model=args.embedding_model,
                        embedding_dims=args.embedding_dims,
                        qdrant_path=args.qdrant_path,
                        history_db_path=args.history_db_path,
                        collection_name=args.collection_name,
                        backend="openai",
                    )
                    search_manager.process_data_file(args.dataset_path)
                _log_phase_duration("mem0 search phase", search_started_at)
        elif args.technique_type == "sage":
            from src.memzero.add_abla import MemoryADDAbla
            from src.memzero.search_local_backend import MemorySearchLocal

            if args.method in ("add", "full"):
                add_started_at = time.perf_counter()
                memory_manager = MemoryADDAbla(
                    data_path=args.dataset_path,
                    batch_size=args.batch_size,
                    is_graph=args.is_graph,
                    stats_output_path=args.action_stats_file,
                    backend=("openai" if args.mem0_backend == "openai" else "ollama"),
                    infer_add=args.infer_add,
                    ollama_base_url=args.ollama_base_url,
                    llm_model=args.llm_model,
                    embedding_model=args.embedding_model,
                    embedding_dims=args.embedding_dims,
                    qdrant_path=args.qdrant_path,
                    history_db_path=args.history_db_path,
                    collection_name=args.collection_name,
                    enable_sage=True,
                )
                memory_manager.process_all_conversations(max_workers=args.max_workers)
                _log_phase_duration("sage add phase", add_started_at)

                # Release the Qdrant file lock so the search phase can open the same path
                if hasattr(memory_manager, "memory") and memory_manager.memory is not None:
                    if hasattr(memory_manager.memory, "vector_store"):
                        vs = memory_manager.memory.vector_store
                        if hasattr(vs, "client") and hasattr(vs.client, "close"):
                            vs.client.close()
                    if hasattr(memory_manager.memory, "db"):
                        memory_manager.memory.db.close()
                del memory_manager

            if args.method in ("search", "full"):
                search_started_at = time.perf_counter()
                output_file_path = os.path.join(
                    args.output_folder,
                    f"sage_{args.llm_model}_vmf_{run_stamp}_{os.getpid()}.json",
                )
                search_manager = MemorySearchLocal(
                    output_path=output_file_path,
                    top_k=args.top_k,
                    ollama_base_url=args.ollama_base_url,
                    llm_model=args.llm_model,
                    embedding_model=args.embedding_model,
                    embedding_dims=args.embedding_dims,
                    qdrant_path=args.qdrant_path,
                    history_db_path=args.history_db_path,
                    collection_name=args.collection_name,
                    backend=("openai" if args.mem0_backend == "openai" else "ollama"),
                )
                search_manager.process_data_file(args.dataset_path)
                _log_phase_duration("sage search phase", search_started_at)
        elif args.technique_type == "rag":
            from src.rag import RAGManager

            rag_started_at = time.perf_counter()
            output_file_path = os.path.join(args.output_folder, f"rag_results_{args.chunk_size}_k{args.num_chunks}.json")
            rag_manager = RAGManager(data_path="dataset/locomo10_rag.json", chunk_size=args.chunk_size, k=args.num_chunks)
            rag_manager.process_all_conversations(output_file_path)
            _log_phase_duration("rag run", rag_started_at)
        elif args.technique_type == "langmem":
            from src.langmem import LangMemManager

            langmem_started_at = time.perf_counter()
            output_file_path = os.path.join(args.output_folder, "langmem_results.json")
            langmem_manager = LangMemManager(dataset_path="dataset/locomo10_rag.json")
            langmem_manager.process_all_conversations(output_file_path)
            _log_phase_duration("langmem run", langmem_started_at)
        elif args.technique_type == "zep":
            from src.zep.add import ZepAdd
            from src.zep.search import ZepSearch

            if args.method == "add":
                zep_add_started_at = time.perf_counter()
                zep_manager = ZepAdd(data_path=args.dataset_path)
                zep_manager.process_all_conversations("1")
                _log_phase_duration("zep add phase", zep_add_started_at)
            elif args.method == "search":
                zep_search_started_at = time.perf_counter()
                output_file_path = os.path.join(args.output_folder, "zep_search_results.json")
                zep_manager = ZepSearch()
                zep_manager.process_data_file(args.dataset_path, "1", output_file_path)
                _log_phase_duration("zep search phase", zep_search_started_at)
        elif args.technique_type == "openai":
            from src.openai.predict import OpenAIPredict

            openai_started_at = time.perf_counter()
            output_file_path = os.path.join(args.output_folder, "openai_results.json")
            openai_manager = OpenAIPredict()
            openai_manager.process_data_file(args.dataset_path, output_file_path)
            _log_phase_duration("openai run", openai_started_at)
        else:
            raise ValueError(f"Invalid technique type: {args.technique_type}")
    finally:
        script_elapsed_seconds = time.perf_counter() - script_started_at
        script_finished_wall = datetime.now()
        script_elapsed_readable = _format_elapsed_time(script_elapsed_seconds)
        script_timing_message = (
            f"[TIMER] Script started at {script_started_wall.isoformat(timespec='seconds')}, "
            f"finished at {script_finished_wall.isoformat(timespec='seconds')}, "
            f"total runtime {script_elapsed_readable}"
        )
        print(script_timing_message)
        logging.info(script_timing_message)


if __name__ == "__main__":
    main()
