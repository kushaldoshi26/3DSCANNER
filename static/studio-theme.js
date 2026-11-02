// Presentation layer only: keep the scanner's existing navigation and API handlers.
document.addEventListener('DOMContentLoaded', () => {
    const intro = document.getElementById('studioIntro');
    const update = () => {
        const workspace = document.getElementById('navWorkspace').classList.contains('active');
        intro.hidden = !workspace;
        document.querySelectorAll('[data-studio-nav]').forEach(button => {
            const target = document.getElementById(button.dataset.studioNav);
            button.classList.toggle('active', !!target?.classList.contains('active'));
        });
    };
    document.querySelectorAll('[data-studio-nav]').forEach(button => {
        button.addEventListener('click', event => {
            event.preventDefault();
            let target = document.getElementById(button.dataset.studioNav);
            if (target.id === 'navProfile' && target.style.display === 'none') target = document.getElementById('navAuth');
            target.click();
        });
    });
    new MutationObserver(update).observe(document.querySelector('.header-tools'), {subtree:true, attributes:true, attributeFilter:['class']});
    update();
});

window.addEventListener('message', event => {
    const frame = document.getElementById('scannerHomeFrame');
    if (event.origin !== location.origin || event.source !== frame?.contentWindow || event.data?.type !== 'scanner-navigation') return;
    const allowed = ['navHome','navWorkspace','navGallery','navProfile','navAuth'];
    if (!allowed.includes(event.data.target)) return;
    let target = document.getElementById(event.data.target);
    if (target.id === 'navProfile' && target.style.display === 'none') target = document.getElementById('navAuth');
    target.click();
});
