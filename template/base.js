
(function(){
  "use strict";

  /* ---- Persist: localStorage with graceful in-memory fallback ---- */
  var Persist = (function(){
    var mem = {}, ok = true;
    try { var k="__ls_test__"; localStorage.setItem(k,"1"); localStorage.removeItem(k); }
    catch(e){ ok = false; }
    return {
      get: function(key){ try { return ok ? localStorage.getItem(key) : (mem[key]||null); } catch(e){ return mem[key]||null; } },
      set: function(key,val){ try { if(ok) localStorage.setItem(key,val); else mem[key]=val; } catch(e){ mem[key]=val; } }
    };
  })();

  /* ---- Copy buttons (innerText so highlight spans don't leak) ---- */
  document.querySelectorAll('.code').forEach(function(block){
    var btn = block.querySelector('.copy'), pre = block.querySelector('pre');
    if(!btn || !pre) return;
    btn.addEventListener('click', function(){
      var text = pre.innerText;
      navigator.clipboard.writeText(text).then(function(){
        btn.textContent='Copied'; btn.classList.add('done');
        setTimeout(function(){ btn.textContent='Copy'; btn.classList.remove('done'); }, 1500);
      }).catch(function(){ btn.textContent='Press ⌘C'; });
    });
  });

  /* ---- Scroll-spy TOC ---- */
  var links = Array.prototype.slice.call(document.querySelectorAll('nav.toc a'));
  var map = {};
  links.forEach(function(a){ map[a.getAttribute('href').slice(1)] = a; });
  if('IntersectionObserver' in window){
    var obs = new IntersectionObserver(function(entries){
      entries.forEach(function(e){
        if(e.isIntersecting){
          links.forEach(function(a){ a.classList.remove('active'); });
          var a = map[e.target.id]; if(a) a.classList.add('active');
        }
      });
    }, { rootMargin: '-12% 0px -75% 0px', threshold: 0 });
    document.querySelectorAll('main section[id]').forEach(function(s){ obs.observe(s); });
  }

  /* ---- Theme dropdown ---- */
  var THEMES = ['candy','retro','noir','beige'];
  var themeBtn = document.getElementById('themeBtn'), themeList = document.getElementById('themeList');
  function applyTheme(t){
    if(THEMES.indexOf(t) < 0) t = 'candy';
    document.documentElement.dataset.theme = t;
    Persist.set('qft-theme', t);
    themeList.querySelectorAll('button[data-theme]').forEach(function(b){
      b.setAttribute('aria-current', b.dataset.theme === t ? 'true' : 'false');
    });
  }
  function openMenu(open){ themeList.hidden = !open; themeBtn.setAttribute('aria-expanded', open?'true':'false'); }
  themeBtn.addEventListener('click', function(e){ e.stopPropagation(); openMenu(themeList.hidden); });
  themeList.querySelectorAll('button[data-theme]').forEach(function(b){
    b.addEventListener('click', function(){ applyTheme(b.dataset.theme); openMenu(false); themeBtn.focus(); });
  });
  document.addEventListener('click', function(){ if(!themeList.hidden) openMenu(false); });
  document.addEventListener('keydown', function(e){ if(e.key==='Escape' && !themeList.hidden){ openMenu(false); themeBtn.focus(); } });
  applyTheme(Persist.get('qft-theme') || 'candy');

  /* ---- Glossary tooltips (click/Enter/Space to pin; for touch) ---- */
  document.querySelectorAll('.gloss').forEach(function(g){
    g.setAttribute('aria-expanded', 'false');                 // announce disclosure state to screen readers
    function sync(){ g.setAttribute('aria-expanded', g.classList.contains('open') ? 'true' : 'false'); }
    g.addEventListener('click', function(e){ e.stopPropagation(); g.classList.toggle('open'); sync(); });
    g.addEventListener('keydown', function(e){ if(e.key==='Enter'||e.key===' '){ e.preventDefault(); g.classList.toggle('open'); sync(); } });
  });
  document.addEventListener('click', function(){ document.querySelectorAll('.gloss.open').forEach(function(g){ g.classList.remove('open'); g.setAttribute('aria-expanded','false'); }); });

