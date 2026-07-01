
document.addEventListener('DOMContentLoaded', () => {
const sourceText=document.getElementById('source-text');
const targetText=document.getElementById('target-text');
const sourceLang=document.getElementById('source-lang');
const targetLang=document.getElementById('target-lang');
const translateBtn=document.getElementById('translate-btn');
const swapBtn=document.getElementById('swap-langs');
const copyBtn=document.getElementById('copy-target');
const speakSourceBtn=document.getElementById('speak-source');
const speakTargetBtn=document.getElementById('speak-target');
const charCount=document.getElementById('char-count');

function updateCount(){charCount.textContent=`${sourceText.value.length} / 1000`;}
sourceText.addEventListener('input',updateCount);

swapBtn.addEventListener('click',()=>{
[sourceLang.value,targetLang.value]=[targetLang.value,sourceLang.value];
[sourceText.value,targetText.value]=[targetText.value,sourceText.value];
updateCount();
});

translateBtn.addEventListener('click',async()=>{
const text=sourceText.value.trim();
if(!text){targetText.value='';return;}
translateBtn.disabled=true;
targetText.value='Translating...';
try{
const url=`https://api.mymemory.translated.net/get?q=${encodeURIComponent(text)}&langpair=${sourceLang.value}|${targetLang.value}`;
const res=await fetch(url);
if(!res.ok) throw new Error();
const data=await res.json();
targetText.value=data.responseData?.translatedText || 'Translation unavailable.';
}catch{
targetText.value='Unable to translate. Check your internet connection and try again.';
}finally{translateBtn.disabled=false;}
});

copyBtn.addEventListener('click',async()=>{
if(!targetText.value||targetText.value==='Translating...') return;
try{
await navigator.clipboard.writeText(targetText.value);
const i=copyBtn.querySelector('i');
i.className='fas fa-check';
setTimeout(()=>i.className='fas fa-copy',1500);
}catch(e){}
});

function speak(text,lang){
if(!text||text==='Translating...') return;
speechSynthesis.cancel();
const u=new SpeechSynthesisUtterance(text);
u.lang=lang;
speechSynthesis.speak(u);
}
speakSourceBtn.onclick=()=>speak(sourceText.value,sourceLang.value);
speakTargetBtn.onclick=()=>speak(targetText.value,targetLang.value);
updateCount();
});
