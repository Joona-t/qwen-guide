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
  window.FineTuneMath = Object.freeze({sigmoid:sigmoid, loss:loss, adapter:adapter, weights:weights, QUIZ:QUIZ,
    relu:relu, wiredProgram:wiredProgram, wiredGate:wiredGate, wiredNetwork:wiredNetwork});

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
