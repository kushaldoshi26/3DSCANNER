const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
function element(){
    const events={},attrs={},classes=new Set();
    return {events,attrs,dataset:{},value:'',type:'password',tagName:'INPUT',disabled:false,
        classList:{contains:c=>classes.has(c),add:c=>classes.add(c),remove:c=>classes.delete(c)},
        addEventListener(name,fn){(events[name]??=[]).push(fn);},
        setAttribute(k,v){attrs[k]=v;},after(){},
        fire(name,event={}){for(const fn of events[name]||[])fn(event);},
        click(){this.clicked=(this.clicked||0)+1;this.fire('click');}
    };
}
const ids=Object.fromEntries(['tabLogin','tabRegister','authHeading','authDescription','authMessage','loginForm','registerForm','btnSubmitLogin','btnSubmitRegister','regPass','regPassConfirm'].map(id=>[id,element()]));
ids.tabLogin.classList.add('active');
ids.regPass.parentElement=element();ids.regPassConfirm.parentElement=element();
const toggle=element(),input=element(),forgot=element(),created=[];
toggle.parentElement={querySelector:()=>input};
toggle.addEventListener('click',()=>{input.type=input.type==='password'?'text':'password';});
let boot,message;
const document={getElementById:id=>ids[id],addEventListener:(name,fn)=>{boot=fn;},
    querySelectorAll:()=>[toggle],querySelector:()=>forgot,
    createElement:()=>{const e=element();created.push(e);return e;}};
vm.runInNewContext(fs.readFileSync('static/auth-polish.js','utf8'),{document,showAuthMessage:text=>{message=text;}});
boot();
assert.equal(ids.authHeading.textContent,'Welcome back.');
let prevented=0;
const key={key:'Enter',target:{tagName:'INPUT'},preventDefault(){prevented++;}};
ids.loginForm.fire('keydown',key);
assert.equal(ids.btnSubmitLogin.clicked,1);
ids.btnSubmitLogin.disabled=true;ids.loginForm.fire('keydown',key);
assert.equal(ids.btnSubmitLogin.clicked,1,'disabled submissions must not duplicate');
ids.loginForm.fire('keydown',{...key,isComposing:true});assert.equal(prevented,2);
ids.tabRegister.classList.add('active');ids.tabLogin.classList.remove('active');ids.tabRegister.click();
assert.equal(ids.authHeading.textContent,'Make room for your ideas.');
assert.equal(ids.tabRegister.attrs['aria-selected'],'true');
ids.regPass.value='example-password';ids.regPassConfirm.value='different';ids.regPassConfirm.fire('input');
assert.equal(created[1].textContent,'Passwords do not match yet.');
ids.regPassConfirm.value=ids.regPass.value;ids.regPassConfirm.fire('input');
assert.equal(created[1].textContent,'Passwords match.');
toggle.fire('keydown',{key:' ',preventDefault(){},stopPropagation(){}});
assert.equal(input.type,'text');assert.equal(toggle.attrs['aria-label'],'Hide password');
forgot.fire('click',{preventDefault(){}});assert.match(message,/not configured/);
console.log('Account keyboard submission, duplicate prevention, password matching, accessible visibility controls and recovery feedback passed.');
