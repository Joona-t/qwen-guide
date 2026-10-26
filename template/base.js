(function(){
  "use strict";

  /* ---- Persist: localStorage with an in-memory fallback ---- */
  var Persist = (function(){
    var mem = {}, ok = false;
    try { localStorage.setItem('__t','1'); localStorage.removeItem('__t'); ok = true; } catch(e) {}
    return {
      get: function(k){ try { return ok ? localStorage.getItem(k) : (mem[k] || null); } catch(e){ return mem[k] || null; } },
      set: function(k,v){ try { if(ok) localStorage.setItem(k,v); else mem[k]=v; } catch(e){ mem[k]=v; } }
    };
  })();
  var root = document.documentElement;

  /* ---- Appearance menu: theme + reading font ---- */
  var btn = document.getElementById('themeBtn'), menu = document.getElementById('themeMenu');
  function apply(kind, val){
    root.setAttribute('data-' + kind, val);
    Persist.set('qft-' + kind, val);
    menu.querySelectorAll('button[data-' + kind + ']').forEach(function(b){
      b.setAttribute('aria-checked', b.getAttribute('data-' + kind) === val ? 'true' : 'false');
    });
  }
  function openMenu(open){ menu.hidden = !open; btn.setAttribute('aria-expanded', open ? 'true' : 'false'); if(open){ var f = menu.querySelector('[aria-checked="true"]') || menu.querySelector('button'); f && f.focus(); } }
  btn.addEventListener('click', function(e){ e.stopPropagation(); openMenu(menu.hidden); });
  menu.addEventListener('click', function(e){ e.stopPropagation(); });
  menu.querySelectorAll('button[data-theme]').forEach(function(b){ b.addEventListener('click', function(){ apply('theme', b.dataset.theme); }); });
  menu.querySelectorAll('button[data-font]').forEach(function(b){ b.addEventListener('click', function(){ apply('font', b.dataset.font); }); });
  document.addEventListener('click', function(){ if(!menu.hidden) openMenu(false); });
  document.addEventListener('keydown', function(e){ if(e.key === 'Escape' && !menu.hidden){ openMenu(false); btn.focus(); } });
  // 'candy' was renamed 'bubblegum' on 2026-10-25; apply() re-saves the new name.
  var savedTheme = Persist.get('qft-theme');
  // Basalt (lovespark.love's look) is the default since 2026-10-27; its Serif setting reads in
  // Cinzel. Saved choices win, so returning readers keep their theme and font.
  apply('theme', savedTheme === 'candy' ? 'bubblegum' : (savedTheme || 'basalt'));
  apply('font', Persist.get('qft-font') || 'serif');

  /* ---- Copy buttons on code blocks ---- */
  document.querySelectorAll('.code').forEach(function(block){
    var b = block.querySelector('.copy'), pre = block.querySelector('pre');
    if(!b || !pre) return;
    b.addEventListener('click', function(){
      var text = pre.innerText;
      (navigator.clipboard ? navigator.clipboard.writeText(text) : Promise.reject()).then(function(){
        b.textContent = 'Copied'; b.classList.add('done');
        setTimeout(function(){ b.textContent = 'Copy'; b.classList.remove('done'); }, 1500);
      }).catch(function(){ b.textContent = 'Press ⌘C'; });
    });
  });

  /* ---- Contents rail: highlight the section in view ---- */
  var links = Array.prototype.slice.call(document.querySelectorAll('.toc a'));
  if('IntersectionObserver' in window && links.length){
    var map = {}; links.forEach(function(a){ map[a.getAttribute('href').slice(1)] = a; });
    var obs = new IntersectionObserver(function(entries){
      entries.forEach(function(e){
        if(e.isIntersecting && map[e.target.id]){
          links.forEach(function(a){ a.classList.remove('active'); a.removeAttribute('aria-current'); });
          map[e.target.id].classList.add('active'); map[e.target.id].setAttribute('aria-current', 'true');
        }
      });
    }, { rootMargin: '-20% 0px -70% 0px' });
    document.querySelectorAll('main section[id]').forEach(function(s){ obs.observe(s); });
  }

  /* ---- Reading progress bar ---- */
  var bar = document.getElementById('progress');
  if(bar){
    var tick = false;
    window.addEventListener('scroll', function(){
      if(tick) return; tick = true;
      requestAnimationFrame(function(){
        var h = document.documentElement.scrollHeight - innerHeight;
        bar.style.transform = 'scaleX(' + (h > 0 ? Math.min(1, scrollY / h) : 0) + ')'; tick = false;
      });
    }, { passive: true });
  }

  /* ---- Expand wide tables beyond the reading column ---- */
  var expanded = [];
  function sizeExpanded(wrap){
    wrap.style.width = ''; wrap.style.marginLeft = '';
    var g = Math.max(16, innerWidth * 0.055), left = wrap.getBoundingClientRect().left;
    var toc = document.querySelector('.toc'), railVisible = toc && getComputedStyle(toc).display !== 'none';
    if(railVisible){ wrap.style.width = Math.max(wrap.clientWidth, innerWidth - g - left) + 'px'; }
    else { wrap.style.width = (innerWidth - 2 * g) + 'px'; wrap.style.marginLeft = (g - left) + 'px'; }
  }
  document.querySelectorAll('.tbl-expand').forEach(function(b){
    var wrap = b.closest('.tablewrap');
    b.addEventListener('click', function(){
      var on = wrap.classList.toggle('expanded');
      b.setAttribute('aria-expanded', on ? 'true' : 'false');
      b.textContent = on ? 'Collapse table' : 'Expand table';
      if(on){ expanded.push(wrap); sizeExpanded(wrap); if(wrap.getBoundingClientRect().top < 0) wrap.scrollIntoView({ block: 'start' }); }
      else { expanded = expanded.filter(function(w){ return w !== wrap; }); wrap.style.width = ''; wrap.style.marginLeft = ''; }
    });
  });
  window.addEventListener('resize', function(){ expanded.forEach(sizeExpanded); });

