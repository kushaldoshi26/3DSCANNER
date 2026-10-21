import './navigation.css';
// This published Site is owner-private: dispatch requires a real signed-in
// visitor before serving any HTML. No client-stored login or passwords are used.
export const signedIn = location.hostname === 'form-scanner-electronics.zippyshell32.chatgpt.site';
export function navigation(current='home') {
 const items=[['home','./','Home'],['workspace','./workspace.html','Workspace'],['about','./about.html','About']];
 const account=signedIn?[['profile','./profile.html','Profile']]:[['login','./login.html','Login'],['register','./register.html','Register']];
 return `<nav class="scanner-nav" aria-label="Main navigation">${[...items,...account].map(([id,url,label])=>`<a href="${url}" ${current===id?'aria-current="page"':''} class="${id==='register'?'nav-register':''}">${label}</a>`).join('')}</nav>`;
}
export const brand='<a class="scanner-brand" href="./">3D SCANNER</a>';
export const signInPath='/signin-with-chatgpt?return_to=%2Fprofile.html';
export const signOutPath='/signout-with-chatgpt?return_to=%2F';
