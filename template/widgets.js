  /* ---- Fine-tuning widgets (pure math exposed as FineTuneMath for scripts/verify.mjs) ---- */
  var $ = function(id){ return document.getElementById(id); };
  var sigmoid = function(w){ return 1/(1+Math.exp(-w)); };
  var loss = function(w,y){ return Math.max(w,0) - w*y + Math.log1p(Math.exp(-Math.abs(w))); };
  var adapter = function(din,dout,r,n){ return {base:din*dout*n, trainable:r*(din+dout)*n, ratio:r*(din+dout)/(din*dout)*100}; };
  var weights = function(b,bits){ return {gb:b*bits/8, gib:b*1e9*bits/8/Math.pow(2,30)}; };
  var clamp = function(v,lo,hi,fb){ var n=Number(v); return isFinite(n)&&n>=lo&&n<=hi ? n : fb; };
  var QUIZ = ['b','b','a','b'];
  /* handwired: the program `a > 4 ? 10 : 12` and its exact rewrite as weights + ReLU (after chenglou/handwired, MIT) */
  var relu = function(x){ return Math.max(0,x); };
  var wiredProgram = function(a){ return a>4 ? 10 : 12; };
  var wiredGate = function(a){ return relu(a-4) - relu(a-5); };
  var wiredNetwork = function(a){ var g=wiredGate(a); return g*10 + (1-g)*12; };
  /* ported from the retired qwen-local-guide: decode ceiling, 16 GB budget bands, compounding reliability */
  var decode = function(bw,gb){ return bw/gb; };
  var BUDGET = {os:4, usableLow:9, usableHigh:12, total:16, rag:1.2};
  var budget = function(model,rag,longCtx,fa){
    var kv = longCtx ? (fa ? 1.5 : 2) : 1, w = model + (rag ? BUDGET.rag : 0) + kv;
    return {kv:kv, rag:rag ? BUDGET.rag : 0, workload:w, verdict: w<=BUDGET.usableLow ? 'ok' : w<=BUDGET.usableHigh ? 'tight' : 'over'};
  };
  var chain = function(p,n){ return Math.pow(p,n); };
  /* fit + decode ceiling (research.md §2): usable = VRAM − ~1 GB per card, or ~75% of Mac unified memory; KV only where the per-token constant is verified */
  var FIT = {gpuReserve:1, macShare:0.75, tightShare:0.9, q8:0.5, unknownKvHeadroom:5};   // with no verified KV constant, under 5 GB of headroom counts as tight
  var decodeCeiling = function(bw,gb){ return gb>0 ? bw/gb : 0; };
  var fitCalc = function(o){
    var cards=o.cards||1, usable = o.kind==='mac' ? o.mem*FIT.macShare : o.mem - FIT.gpuReserve*cards;
    var kv = o.kvKiB ? o.ctx*o.kvKiB*1024*(o.kvType==='q8_0' ? FIT.q8 : 1)/1e9 : null;
    var need = o.model + (kv||0);
    return {usable:usable, reserved:o.mem-usable, weights:o.model, kv:kv, need:need,
      verdict: need>usable ? 'no' : (kv===null ? (usable-need>=FIT.unknownKvHeadroom ? 'fits' : 'tight') : (need<=usable*FIT.tightShare ? 'fits' : 'tight'))};
  };
  /* quant widget (research.md §B, §C, §E): real GGUF sizes (GB, checked 2026-11-03), params from model.safetensors totals, KV KiB/token at f16 from config.json */
  var QMODELS = {
    q9:   {name:'Qwen3.5-9B', params:9.65, active:9.65, kvKiB:32, quants:[['UD-IQ2_XXS',3.19],['UD-Q3_K_XL',5.05],['Q4_K_M',5.68],['UD-Q4_K_XL',5.97],['Q5_K_M',6.58],['Q6_K',7.46],['Q8_0',9.53],['BF16',17.92]]},
    q27:  {name:'Qwen3.8-27B', params:27.78, active:27.78, kvKiB:64, quants:[['UD-IQ2_XXS',7.3],['UD-Q2_K_XL',9.8],['UD-IQ3_XXS',10.9],['UD-Q3_K_XL',13.1],['UD-IQ4_XS',14.3],['UD-Q4_K_XL',17.6],['Q4_K_M',19.0],['UD-Q5_K_XL',20.9],['UD-Q6_K',22.0],['Q8_0',28.6],['BF16',53.8]]},
    q122: {name:'Qwen3.5-122B-A10B', params:125.09, active:10, kvKiB:24, quants:[['UD-IQ2_XXS',36.6],['UD-IQ3_XXS',44.7],['UD-Q3_K_XL',57.0],['Q4_K_M',76.5],['Q8_0',129.9]]}
  };
  var NATIVE_CTX = 262144;
  var quantCalc = function(o){   // o: {params, active, kvKiB, size, kind, mem, bw, cards, kvType}
    var usable = o.kind==='mac' ? o.mem*FIT.macShare : o.mem - FIT.gpuReserve*o.cards;
    var bpw = o.size*8/o.params, perTok = o.kvKiB*1024*(o.kvType==='q8_0' ? FIT.q8 : 1);
    var left = usable - o.size, ctx = left>0 ? Math.min(NATIVE_CTX, Math.floor(left*1e9/perTok)) : 0;
    return {usable:usable, bpw:bpw, left:left, ctx:ctx, ceiling: o.bw/(o.size*o.active/o.params), fits: left>0};
  };
  window.FineTuneMath = Object.freeze({sigmoid:sigmoid, loss:loss, adapter:adapter, weights:weights, QUIZ:QUIZ,
    relu:relu, wiredProgram:wiredProgram, wiredGate:wiredGate, wiredNetwork:wiredNetwork,
    decode:decode, budget:budget, BUDGET:BUDGET, chain:chain, FIT:FIT, fitCalc:fitCalc, decodeCeiling:decodeCeiling, QMODELS:QMODELS, quantCalc:quantCalc});

  if($('calc-step')){
    var w=-2, steps=0, hist=[];
    var fmtW = function(x){ return (x<0?'−':'')+Math.abs(x).toFixed(4); };
    var draw = function(){
      var y=Number($('target').value), p=sigmoid(w), l=loss(w,y);
      $('wOut').textContent=fmtW(w); $('pOut').textContent=(p*100).toFixed(2)+'%'; $('lossOut').textContent=l.toFixed(4);
      var max=Math.max.apply(null, hist.concat([0.01])), den=Math.max(hist.length-1,1);
      var pts=hist.map(function(v,i){ return [40+i/den*550, 134-v/max*114]; });
      $('lossLine').setAttribute('points', pts.map(function(q){ return q.join(','); }).join(' '));
      var last=pts[pts.length-1]; $('lossDot').setAttribute('cx',last[0]); $('lossDot').setAttribute('cy',last[1]);
      $('step1').disabled=$('step10').disabled=steps>=500;
    };
    var reset = function(){ w=-2; steps=0; hist=[loss(w,Number($('target').value))];
      $('stepNote').textContent='Reset to w = −2. '+(Number($('target').value)===1?'The correct label pushes w up.':'The wrong label pushes w down. Lower loss here means it is learning the wrong answer.'); draw(); };
    var take = function(n){ var y=Number($('target').value), lr=clamp($('lr').value,0.01,10,0.5), old=w, g=0;
      for(var i=0;i<n&&steps<500;i++){ old=w; g=sigmoid(w)-y; w-=lr*g; steps++; hist.push(loss(w,y)); }
      $('stepNote').textContent='Step '+steps+': '+fmtW(old)+' − '+lr.toFixed(2)+' × '+fmtW(g)+' = '+fmtW(w)+'. Loss '+hist[0].toFixed(4)+' → '+hist[hist.length-1].toFixed(4)+' on this one example.'; draw(); };
    $('step1').addEventListener('click', function(){ take(1); });
    $('step10').addEventListener('click', function(){ take(10); });
    $('stepReset').addEventListener('click', reset);
    $('target').addEventListener('change', reset);
    reset();
  }

  if($('calc-wired')){
    var fmt = function(x){ return Number.isInteger(x) ? String(x) : x.toFixed(2); };
    var updWired = function(){
      var a=clamp($('wiredA').value,0,10,4), p=wiredProgram(a), g=wiredGate(a), n=wiredNetwork(a);
      $('wiredAOut').textContent=$('wiredA1').textContent=$('wiredA2').textContent=fmt(a);
      $('wiredGate').textContent=fmt(g); $('wiredProg').textContent=fmt(p); $('wiredNet').textContent=fmt(n);
      $('wiredNote').textContent = p===n
        ? 'Both say '+fmt(p)+'. The constants 4, 5, 10 and 12 are the weights; nothing was trained.'
        : 'They differ: program '+fmt(p)+', network '+fmt(n)+'. Between 4 and 5 the gate is '+fmt(g)+', not 0 or 1, so the rewrite is exact everywhere except on this ramp.';
    };
    $('wiredA').addEventListener('input',updWired); $('wiredA').addEventListener('change',updWired);
    updWired();
  }

  if($('calc-decode')){
    var updDecode = function(){
      var custom=$('msize').value==='custom'; $('msizeCustomWrap').hidden=!custom;
      var bw=Number($('dev').value), gb=custom ? clamp($('msizeCustom').value,0.5,200,6.6) : Number($('msize').value), t=decode(bw,gb);
      $('decodeOut').innerHTML='~'+Math.round(t)+' <small>tokens/s ceiling</small>';
      $('decodeNote').textContent=bw.toLocaleString('en-US')+' ÷ '+gb+' ≈ '+Math.round(t)+' tokens per second, before overhead. '+(bw<=273 ? 'Plan for roughly '+Math.round(t*0.77)+'–'+Math.round(t*0.88)+'.' : 'GPU bandwidth here is derived; real runs land lower, so measure.');
    };
    ['dev','msize','msizeCustom'].forEach(function(id){ $(id).addEventListener('input',updDecode); $(id).addEventListener('change',updDecode); });
    updDecode();
  }

  if($('calc-budget')){
    var pct = function(gb){ return (gb/BUDGET.total*100).toFixed(2)+'%'; };
    var updBudget = function(){
      var custom=$('bmodel').value==='custom'; $('bcustomWrap').hidden=!custom;
      var m=custom ? clamp($('bcustom').value,0.5,14,9) : Number($('bmodel').value);
      var b=budget(m,$('brag').checked,$('bctx').value==='long',$('bfa').checked);
      $('bsOs').style.width=pct(BUDGET.os); $('bsModel').style.width=pct(m); $('bsRag').style.width=pct(b.rag); $('bsKv').style.width=pct(b.kv);
      $('budgetBar').setAttribute('aria-label','macOS '+BUDGET.os+' GB, model '+m+' GB, retrieval '+b.rag+' GB, KV and runtime '+b.kv+' GB, workload '+b.workload.toFixed(1)+' GB of a 9 to 12 GB usable window');
      $('budgetOut').innerHTML=b.workload.toFixed(1)+' <small>GB workload</small>';
      $('budgetNote').textContent = (b.verdict==='ok' ? 'Fits comfortably: within the conservative '+BUDGET.usableLow+' GB floor.'
        : b.verdict==='tight' ? 'Tight: inside the '+BUDGET.usableLow+'–'+BUDGET.usableHigh+' GB band only at the optimistic end. Keep the context short.'
        : 'Over: above ~'+BUDGET.usableHigh+' GB. Use a smaller model, drop retrieval, or shorten the context.')
        +' Model '+m+' + retrieval '+b.rag+' + KV '+b.kv+' = '+b.workload.toFixed(1)+' GB.';
    };
    ['bmodel','bcustom','bctx','brag','bfa'].forEach(function(id){ $(id).addEventListener('input',updBudget); $(id).addEventListener('change',updBudget); });
    updBudget();
  }

  if($('calc-fit')){
    var updFit = function(){
      var hw=$('fhw').value.split('|'), custom=$('fmodel').value==='custom'; $('fcustomWrap').hidden=!custom;
      var mv=custom ? [clamp($('fcustom').value,0.5,200,19),0,0] : $('fmodel').value.split('|').map(Number);
      var mem=Number(hw[1]), bw=Number(hw[2]), cards=Number(hw[3]), ctx=Number($('fctx').value);
      var r=fitCalc({kind:hw[0], mem:mem, cards:cards, model:mv[0], kvKiB:mv[1]||null, ctx:ctx, kvType:$('fkv').value});
      var total=Math.max(mem, r.reserved+r.need), pc=function(gb){ return (gb/total*100).toFixed(2)+'%'; };
      $('fsRes').style.width=pc(r.reserved); $('fsModel').style.width=pc(r.weights); $('fsKv').style.width=pc(r.kv||0);
      var kvTxt = r.kv===null ? 'KV estimate unavailable for this model; leave headroom' : 'KV '+r.kv.toFixed(2)+' GB';
      $('fitBar').setAttribute('aria-label','Reserved '+r.reserved.toFixed(1)+' GB, weights '+r.weights+' GB, '+kvTxt+', of '+mem+' GB');
      $('fitOut').innerHTML = r.verdict==='fits' ? 'Fits' : r.verdict==='tight' ? 'Tight' : 'Doesn’t fit';
      $('fitNote').textContent='Weights '+r.weights+' GB + '+kvTxt+' = '+r.need.toFixed(1)+' GB vs '+r.usable.toFixed(1)+' GB usable ('+mem+' GB minus '+r.reserved.toFixed(1)+' GB reserved).'
        + (r.verdict==='tight' ? (r.kv===null ? ' Less than 5 GB is left for the KV cache, whose size this calculator doesn\u2019t know for this model: keep the context modest and use q8_0 KV.' : ' Under the limit by less than 10%: shorten the context or use q8_0 KV.') : r.verdict==='no' ? ' Pick a smaller file, a lower quant, or more memory.' : '');
      var t=decodeCeiling(bw,mv[0]);
      $('fitSpeed').textContent='Decode ceiling ≈ '+Math.round(t)+' tokens/s ('+bw.toLocaleString('en-US')+' GB/s ÷ '+mv[0]+' GB, derived). Real speed is lower; see Benchmark your machine.'
        + (mv[2] ? ' This is a MoE model: each token reads only the active experts, so measured speed can beat this dense ceiling.' : '')
        + (cards>1 ? ' Two cards in layer split take turns per token, so the ceiling uses one card’s bandwidth.' : '');
    };
    ['fhw','fmodel','fcustom','fctx','fkv'].forEach(function(id){ $(id).addEventListener('input',updFit); $(id).addEventListener('change',updFit); });
    updFit();
  }

  if($('calc-quant')){
    var fillQuants = function(){
      var m=QMODELS[$('qmodel').value], keep=$('qquant').value;
      $('qquant').innerHTML = m.quants.map(function(q){ return '<option value="'+q[0]+'">'+q[0]+', '+q[1]+' GB</option>'; }).join('');
      var pick = m.quants.some(function(q){ return q[0]===keep; }) ? keep : 'Q4_K_M'; $('qquant').value = pick;
    };
    var updQuant = function(){
      var m=QMODELS[$('qmodel').value], q=m.quants.filter(function(x){ return x[0]===$('qquant').value; })[0], hw=$('qhw').value.split('|');
      var base={params:m.params, active:m.active, kvKiB:m.kvKiB, size:q[1], kind:hw[0], mem:Number(hw[1]), bw:Number(hw[2]), cards:Number(hw[3])};
      var f=quantCalc(Object.assign({kvType:'f16'},base)), h=quantCalc(Object.assign({kvType:'q8_0'},base));
      var k=function(n){ return n>=1000 ? Math.round(n/1000)+'K' : String(n); };   // thousands of tokens, as in the prose
      $('qOut').textContent = f.fits ? (f.ctx>=8192 ? 'Fits' : 'Barely fits') : 'Doesn’t fit';
      $('qBits').textContent = m.name+' '+q[0]+': '+q[1]+' GB, about '+f.bpw.toFixed(1)+' bits per weight ('+q[1]+' GB × 8 ÷ '+m.params+'B).';
      $('qSpeed').textContent = f.fits ? 'Decode ceiling ≈ '+Math.round(f.ceiling)+' tokens/s ('+Number(hw[2]).toLocaleString('en-US')+' GB/s ÷ '+(m.active<m.params ? 'the active share of ' : '')+q[1]+' GB, derived). Real speed is lower.' : 'Needs '+(-f.left).toFixed(1)+' GB more than the '+f.usable.toFixed(1)+' GB usable. Pick a lower quant, more memory, or keep MoE experts in system RAM (see "Bigger than 27B").';
      $('qCtx').textContent = f.fits ? 'Context that fits: about '+k(f.ctx)+' tokens with an f16 KV cache, '+k(h.ctx)+' with q8_0 ('+f.left.toFixed(1)+' GB left of '+f.usable.toFixed(1)+' GB usable, '+m.kvKiB+' KiB per token at f16).' : '';
    };
    $('qmodel').addEventListener('change', function(){ fillQuants(); updQuant(); });
    ['qquant','qhw'].forEach(function(id){ $(id).addEventListener('change',updQuant); $(id).addEventListener('input',updQuant); });
    fillQuants(); updQuant();
  }

  if($('calc-rel')){
    var updRel = function(){
      var p=Number($('pcall').value)/100, n=Math.round(clamp($('steps').value,1,20,8)), c=chain(p,n)*100;
      $('stepsVal').textContent=n;
      $('relOut').innerHTML='~'+Math.round(c)+'% <small>succeed end to end</small>';
      $('relBar').style.width=Math.max(1,c)+'%';
      $('relNote').textContent=Math.round(p*1000)/10+'% per call over '+n+' step'+(n===1?'':'s')+' lands about '+Math.round(c)+'% of the time.';
    };
    ['pcall','steps'].forEach(function(id){ $(id).addEventListener('input',updRel); $(id).addEventListener('change',updRel); });
    updRel();
  }

  if($('calc-lora')){
    var updLora = function(){
      var di=Number($('din').value), d0=Number($('dout').value), r=Math.round(clamp($('rank').value,1,256,16)), n=Math.round(clamp($('nmat').value,1,2048,1));
      var a=adapter(di,d0,r,n);
      $('loraOut').innerHTML=a.trainable.toLocaleString('en-US')+' <small>trainable</small>';
      $('loraNote').textContent='vs '+a.base.toLocaleString('en-US')+' frozen · '+a.ratio.toFixed(2)+'% of the adapted matrices · A: '+r+'×'+di+', B: '+d0+'×'+r;
    };
    ['din','dout','rank','nmat'].forEach(function(id){ $(id).addEventListener('input',updLora); $(id).addEventListener('change',updLora); });
    updLora();
  }

  if($('calc-mem')){
    var updMem = function(){
      var custom=$('params').value==='custom'; $('paramsCustomWrap').hidden=!custom;
      var b = custom ? clamp($('paramsCustom').value,0.1,200,7) : Number($('params').value), bits=Number($('bits').value), m=weights(b,bits);
      $('memOut').innerHTML=m.gb.toFixed(2)+' <small>GB of weights</small>';
      $('memNote').textContent='≈ '+m.gib.toFixed(2)+' GiB · '+b+'B × '+bits+' bits ÷ 8';
    };
    ['params','paramsCustom','bits'].forEach(function(id){ $(id).addEventListener('input',updMem); $(id).addEventListener('change',updMem); });
    updMem();
  }

  if($('quizForm')){
    var why=['Loss measures fit to the training examples, not improvement on new inputs.','Weights are only the start; training memory must be measured.','Missing facts are a retrieval problem; fine-tuning teaches patterns, not reliable facts.','A template or end-of-sequence mismatch is the usual cause of gibberish after export.'];
    $('quizForm').addEventListener('submit', function(e){ e.preventDefault();
      var d=new FormData($('quizForm')), ans=[1,2,3,4].map(function(i){ return d.get('q'+i); });
      if(ans.some(function(x){ return x===null; })){ $('quizResult').textContent='Answer all four questions first.'; return; }
      var score=ans.filter(function(a,i){ return a===QUIZ[i]; }).length;
      $('quizResult').textContent=score+'/4 correct. '+why.map(function(t,i){ return (i+1)+': '+(ans[i]===QUIZ[i]?'✓ ':'✗ ')+t; }).join(' ');
    });
    $('quizForm').addEventListener('reset', function(){ $('quizResult').textContent='Answer all four, then check.'; });
  }
