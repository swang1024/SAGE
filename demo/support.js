/*
 * support.js — minimal standalone runtime for the SAGE novelty-gate demo.
 *
 * The demo page (sage_novelty_gate_demo.html) was exported from a design-canvas
 * tool that expected a proprietary "DC" runtime to be served alongside it. That
 * runtime was never published, so on its own the page renders the static markup
 * but nothing animates: the browser ignores the <script type="text/x-dc"> block,
 * and the ref="{{ ... }}" / onClick="{{ ... }}" template bindings are never wired.
 *
 * This shim implements exactly the small subset of that runtime the demo needs:
 *   - a global React.createRef (the only React API the component uses),
 *   - a DCLogic base class for `class Component extends DCLogic`,
 *   - a bootstrap that hoists <helmet> into <head>, evaluates the x-dc script,
 *     instantiates the component, binds its refs/handlers to the existing DOM,
 *     and calls componentDidMount().
 *
 * No external dependencies; safe to serve as a plain static file (GitHub Pages).
 */
(function () {
  'use strict';

  // The component only ever calls React.createRef(); a tiny shim is enough.
  window.React = window.React || { createRef: function () { return { current: null }; } };

  // Base class so `class Component extends DCLogic` resolves. The component
  // defines its own lifecycle methods; the base only needs to exist and carry
  // a props bag.
  function DCLogic() {}
  DCLogic.prototype.props = null;
  window.DCLogic = DCLogic;

  function mustacheName(value) {
    // ref="{{ wrapRef }}" -> "wrapRef"
    var m = /\{\{\s*([\w$]+)\s*\}\}/.exec(value || '');
    return m ? m[1] : null;
  }

  function applyHover(el, hoverCss) {
    // style-hover="background:#5fe2f2" — swap declarations on hover, restore on leave.
    var decls = hoverCss.split(';').map(function (s) { return s.trim(); }).filter(Boolean)
      .map(function (d) {
        var i = d.indexOf(':');
        return [d.slice(0, i).trim(), d.slice(i + 1).trim()];
      });
    var prev = [];
    el.addEventListener('mouseenter', function () {
      prev = decls.map(function (kv) { return [kv[0], el.style.getPropertyValue(kv[0])]; });
      decls.forEach(function (kv) { el.style.setProperty(kv[0], kv[1]); });
    });
    el.addEventListener('mouseleave', function () {
      prev.forEach(function (kv) { el.style.setProperty(kv[0], kv[1]); });
    });
  }

  function boot() {
    var xdc = document.querySelector('x-dc');
    if (!xdc) return;

    // 1. Hoist <helmet> contents (font links + <style>) into <head> so the
    //    page background and typography take effect.
    var helmet = xdc.querySelector('helmet');
    if (helmet) {
      while (helmet.firstElementChild) document.head.appendChild(helmet.firstElementChild);
      helmet.remove();
    }

    // 2. The mountable content root is the component's top-level div.
    var root = xdc.querySelector(':scope > div');
    if (!root) return;

    // 3. Read default prop values from the x-dc script's data-props.
    var scriptEl = document.querySelector('script[type="text/x-dc"], script[data-dc-script]');
    if (!scriptEl) return;
    var props = {};
    try {
      var schema = JSON.parse(scriptEl.getAttribute('data-props') || '{}');
      Object.keys(schema).forEach(function (k) {
        if (k === '$preview') return;
        if (schema[k] && 'default' in schema[k]) props[k] = schema[k].default;
      });
    } catch (e) { /* fall back to component defaults */ }

    // 4. Evaluate the x-dc script to obtain the Component class. DCLogic and
    //    React are passed in as locals so the class body resolves.
    var src = scriptEl.textContent + '\nreturn Component;';
    var Component;
    try {
      Component = new Function('DCLogic', 'React', src)(DCLogic, window.React);
    } catch (e) {
      console.error('[support.js] failed to evaluate demo component:', e);
      return;
    }

    var inst = new Component();
    inst.props = props;

    // 5. Bind ref="{{ name }}" and onClick="{{ name }}" to the live DOM using
    //    the name->value map the component exposes via renderVals().
    var bindings = (typeof inst.renderVals === 'function') ? inst.renderVals() : {};

    var refEls = [root].concat(Array.prototype.slice.call(root.querySelectorAll('[ref]')));
    refEls.forEach(function (el) {
      var name = el.getAttribute && mustacheName(el.getAttribute('ref'));
      if (name && bindings[name] && typeof bindings[name] === 'object') {
        bindings[name].current = el;
      }
      if (el.removeAttribute) el.removeAttribute('ref');
    });

    var clickEls = [root].concat(Array.prototype.slice.call(root.querySelectorAll('[onClick]')));
    clickEls.forEach(function (el) {
      var name = mustacheName(el.getAttribute('onClick'));
      el.removeAttribute('onClick');
      if (name && typeof bindings[name] === 'function') {
        el.addEventListener('click', bindings[name]);
      }
    });

    Array.prototype.slice.call(root.querySelectorAll('[style-hover]')).forEach(function (el) {
      applyHover(el, el.getAttribute('style-hover'));
      el.removeAttribute('style-hover');
    });

    // 6. Kick off the component.
    if (typeof inst.componentDidMount === 'function') {
      try { inst.componentDidMount(); }
      catch (e) { console.error('[support.js] componentDidMount failed:', e); }
    }
    window.addEventListener('beforeunload', function () {
      if (typeof inst.componentWillUnmount === 'function') inst.componentWillUnmount();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
