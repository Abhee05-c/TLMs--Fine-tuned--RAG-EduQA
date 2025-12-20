document.addEventListener("click", (e) => {
    const container = document.getElementById("star-burst");
    const starCount = 24;  // number of stars in the circular wave
    const radius = 100;    // distance stars move outward

    for (let i = 0; i < starCount; i++) {
        const star = document.createElement("div");
        star.classList.add("star-particle");

        // Set starting position at click
        star.style.left = e.clientX + "px";
        star.style.top = e.clientY + "px";

        // Calculate angle for circular distribution
        const angle = (i / starCount) * 2 * Math.PI;
        const x = Math.cos(angle) * radius + "px";
        const y = Math.sin(angle) * radius + "px";

        // CSS variables for animation
        star.style.setProperty("--x", x);
        star.style.setProperty("--y", y);

        container.appendChild(star);

        // Remove star after animation completes
        setTimeout(() => {
            star.remove();
        }, 800);
    }
});
