// sidebar active link tracking
const sections = document.querySelectorAll("section[id]");
const links = document.querySelectorAll("#sidebar a");

function setActiveLink() {
    let current = "";
    for (const section of sections) {
        const top = section.offsetTop - 60;
        if (window.scrollY >= top) {
            current = section.id;
        }
    }
    for (const link of links) {
        link.classList.remove("active");
        if (link.getAttribute("href") === `#${current}`) {
            link.classList.add("active");
        }
    }
}

window.addEventListener("scroll", setActiveLink);
setActiveLink();

// copy button
function copyCode(btn) {
    const pre = btn.parentElement.querySelector("pre code");
    const text = pre.textContent;
    navigator.clipboard.writeText(text).then(() => {
        btn.textContent = "Copied!";
        btn.classList.add("copied");
        setTimeout(() => {
            btn.textContent = "Copy";
            btn.classList.remove("copied");
        }, 2000);
    });
}

// mobile sidebar toggle
document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
        document.getElementById("sidebar").classList.remove("open");
    }
});
