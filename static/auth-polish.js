document.addEventListener('DOMContentLoaded', () => {
    const byId=id=>document.getElementById(id);
    if(!byId('tabLogin') || !byId('tabRegister')) return;
    const update=()=>{
        const register=byId('tabRegister').classList.contains('active');
        byId('authHeading').textContent=register?'Make room for your ideas.':'Welcome back.';
        byId('authDescription').textContent=register?'Create an account and start your 3D collection.':'Sign in to continue your next creation.';
        for(const id of ['tabLogin','tabRegister']) byId(id).setAttribute('aria-selected',String(byId(id).classList.contains('active')));
    };
    for(const id of ['tabLogin','tabRegister']) byId(id).addEventListener('click',update);
    byId('authMessage')?.setAttribute('role','status');
    byId('authMessage')?.setAttribute('aria-live','polite');
    for(const [panel,submit] of [['loginForm','btnSubmitLogin'],['registerForm','btnSubmitRegister']]) {
        byId(panel)?.addEventListener('keydown',event=>{
            if(event.key==='Enter' && event.target.tagName==='INPUT' && !event.isComposing){
                event.preventDefault();
                if(!byId(submit).disabled) byId(submit).click();
            }
        });
    }
    document.querySelectorAll('.pwd-toggle').forEach(toggle=>{
        const input=toggle.parentElement.querySelector('input');
        toggle.setAttribute('role','button'); toggle.tabIndex=0;
        const label=()=>toggle.setAttribute('aria-label',`${input.type==='password'?'Show':'Hide'} password`);
        toggle.addEventListener('click',label);
        toggle.addEventListener('keydown',event=>{
            if(event.key==='Enter'||event.key===' '){event.preventDefault();event.stopPropagation();toggle.click();}
        }); label();
    });
    const password=byId('regPass'),confirm=byId('regPassConfirm');
    if(password && confirm){
        const hint=document.createElement('p');hint.className='auth-password-hint';hint.id='passwordHint';
        hint.setAttribute('aria-live','polite'); password.parentElement.after(hint);
        const match=document.createElement('p');match.className='auth-password-hint';match.id='passwordMatch';
        match.setAttribute('aria-live','polite');confirm.parentElement.after(match);
        password.setAttribute('aria-describedby',hint.id);confirm.setAttribute('aria-describedby',match.id);
        const check=()=>{
            hint.textContent=password.value.length>=6?'Minimum length met. Use a unique password.':'Use at least 6 characters.';
            match.textContent=!confirm.value?'':password.value===confirm.value?'Passwords match.':'Passwords do not match yet.';
            match.dataset.valid=String(password.value===confirm.value);
        };
        password.addEventListener('input',check);confirm.addEventListener('input',check);check();
    }
    document.querySelector('.forgot-pwd')?.addEventListener('click',event=>{
        event.preventDefault();showAuthMessage('Password recovery is not configured for this local installation. Contact the project administrator.','error');
    });
    update();
});
