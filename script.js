const revealItems = document.querySelectorAll(".reveal");

if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          revealObserver.unobserve(entry.target);
        }
      });
    },
    {
      threshold: 0.14,
      rootMargin: "0px 0px -8% 0px",
    }
  );

  revealItems.forEach((item, index) => {
    item.style.transitionDelay = `${Math.min(index % 5, 4) * 70}ms`;
    revealObserver.observe(item);
  });
} else {
  revealItems.forEach((item) => item.classList.add("is-visible"));
}

document.querySelectorAll(".project-card").forEach((card) => {
  card.addEventListener("pointermove", (event) => {
    const rect = card.getBoundingClientRect();
    const x = ((event.clientX - rect.left) / rect.width - 0.5) * 8;
    const y = ((event.clientY - rect.top) / rect.height - 0.5) * -8;

    card.style.transform = `perspective(1000px) rotateX(${y}deg) rotateY(${x}deg)`;
  });

  card.addEventListener("pointerleave", () => {
    card.style.transform = "";
  });
});

const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
let transitionNavigationTimer = 0;

function resetPageTransition() {
  if (transitionNavigationTimer) {
    window.clearTimeout(transitionNavigationTimer);
    transitionNavigationTimer = 0;
  }

  document.body.classList.remove("page-transition-out");
  document.querySelectorAll(".page-transition-veil, .project-card-transition-clone").forEach((element) => element.remove());
  document.querySelectorAll(".new-project-card.is-card-leaving").forEach((card) => card.classList.remove("is-card-leaving"));
}

window.addEventListener("pagehide", resetPageTransition);
window.addEventListener("pageshow", resetPageTransition);

document.querySelectorAll("a[href]").forEach((link) => {
  link.addEventListener("click", (event) => {
    if (
      event.defaultPrevented ||
      event.metaKey ||
      event.ctrlKey ||
      event.shiftKey ||
      event.altKey ||
      link.target === "_blank" ||
      link.hasAttribute("download") ||
      prefersReducedMotion.matches
    ) {
      return;
    }

    const url = new URL(link.href, window.location.href);
    const isSameOrigin = url.origin === window.location.origin;
    const isHashOnly =
      url.pathname === window.location.pathname &&
      url.search === window.location.search &&
      url.hash;

    if (!isSameOrigin || isHashOnly || !["http:", "https:"].includes(url.protocol)) {
      return;
    }

    if (document.body.classList.contains("page-transition-out")) {
      event.preventDefault();
      return;
    }

    event.preventDefault();
    const rect = link.getBoundingClientRect();
    const originX = event.clientX || rect.left + rect.width / 2;
    const originY = event.clientY || rect.top + rect.height / 2;
    const isProjectCard = link.classList.contains("new-project-card");
    const veil = document.createElement("span");

    veil.className = "page-transition-veil";
    veil.style.setProperty("--transition-x", `${originX}px`);
    veil.style.setProperty("--transition-y", `${originY}px`);
    document.body.appendChild(veil);

    if (isProjectCard) {
      const clone = link.cloneNode(true);
      clone.classList.remove("reveal", "is-visible");
      clone.classList.add("project-card-transition-clone");
      clone.style.setProperty("--card-top", `${rect.top}px`);
      clone.style.setProperty("--card-left", `${rect.left}px`);
      clone.style.setProperty("--card-width", `${rect.width}px`);
      clone.style.setProperty("--card-height", `${rect.height}px`);
      clone.style.transitionDelay = "0ms";
      document.body.appendChild(clone);
      link.classList.add("is-card-leaving");

      requestAnimationFrame(() => {
        clone.classList.add("is-active");
        veil.classList.add("is-active");
      });
    } else {
      requestAnimationFrame(() => {
        veil.classList.add("is-active");
      });
    }

    document.body.classList.add("page-transition-out");
    transitionNavigationTimer = window.setTimeout(() => {
      transitionNavigationTimer = 0;
      window.location.href = url.href;
    }, isProjectCard ? 430 : 260);
  });
});
