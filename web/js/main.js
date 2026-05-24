/* =========================================================
   main.js - 公共交互逻辑
   ========================================================= */

// ---------------- Tab 切换 ----------------
function switchTab(event, tabId) {
    const btn = event.currentTarget;
    const tabContainer = btn.closest('.section') || document;
    tabContainer.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    tabContainer.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    const target = document.getElementById(tabId);
    if (target) target.classList.add('active');
}

// ---------------- 复制 BibTeX ----------------
function copyBibtex() {
    const block = document.getElementById('bibtex-block');
    if (!block) return;
    // 提取除按钮以外的纯文本
    const text = block.innerText.replace(/^\s*📋\s*复制\s*/, '').trim();
    const btn = block.querySelector('.copy-btn');
    if (navigator.clipboard) {
        navigator.clipboard.writeText(text).then(() => {
            const orig = btn.innerText;
            btn.innerText = '✓ 已复制';
            btn.classList.add('copied');
            setTimeout(() => {
                btn.innerText = orig;
                btn.classList.remove('copied');
            }, 1800);
        });
    } else {
        // 兼容旧浏览器
        const ta = document.createElement('textarea');
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
    }
}

// ---------------- 导航当前位置高亮 ----------------
document.addEventListener('DOMContentLoaded', () => {
    const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
    if (navLinks.length === 0) return;

    const sections = Array.from(navLinks).map(a => {
        const id = a.getAttribute('href').slice(1);
        return { id, link: a, section: document.getElementById(id) };
    }).filter(s => s.section);

    function onScroll() {
        const fromTop = window.scrollY + 120;
        let current = sections[0];
        for (const s of sections) {
            if (s.section.offsetTop <= fromTop) current = s;
        }
        navLinks.forEach(a => a.classList.remove('active'));
        if (current) current.link.classList.add('active');
    }

    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();
});

// ---------------- 图片懒加载 fallback ----------------
document.addEventListener('DOMContentLoaded', () => {
    if ('loading' in HTMLImageElement.prototype) return;
    document.querySelectorAll('img[loading="lazy"]').forEach(img => {
        img.src = img.src; // 触发加载
    });
});

// ---------------- Lightbox（图片放大查看） ----------------
document.addEventListener('DOMContentLoaded', () => {
    if (document.getElementById('global-lightbox')) return;
    const lb = document.createElement('div');
    lb.id = 'global-lightbox';
    lb.style.cssText = `
        display:none; position:fixed; inset:0; z-index:999;
        background:rgba(0,0,0,0.92); justify-content:center;
        align-items:center; cursor:zoom-out;
    `;
    lb.innerHTML = '<img style="max-width:95%;max-height:95%;border-radius:6px;box-shadow:0 0 40px rgba(255,255,255,0.1);"/>';
    document.body.appendChild(lb);

    lb.addEventListener('click', () => lb.style.display = 'none');
    document.addEventListener('keydown', e => {
        if (e.key === 'Escape') lb.style.display = 'none';
    });

    // 给所有 figure 中的图片添加点击放大
    document.querySelectorAll('.figure img, .img-card img').forEach(img => {
        img.style.cursor = 'zoom-in';
        img.addEventListener('click', () => {
            lb.querySelector('img').src = img.src;
            lb.style.display = 'flex';
        });
    });
});
