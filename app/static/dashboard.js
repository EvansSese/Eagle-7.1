function setTheme(theme) {
  const isDark = theme === "dark";
  document.body.dataset.theme = isDark ? "dark" : "light";
  document.querySelector("#theme-toggle").setAttribute("aria-pressed", String(isDark));
  document.querySelector("#theme-icon").textContent = isDark ? "☼" : "☾";
  document.querySelector("#theme-label").textContent = isDark ? "Light theme" : "Dark theme";
  document.querySelector('meta[name="theme-color"]').content = isDark ? "#101812" : "#f7f9f7";
  try {
    localStorage.setItem("eagle-theme", isDark ? "dark" : "light");
  } catch {
    // Theme changes still apply for this page if browser storage is unavailable.
  }
}

let initialTheme = "dark";
try {
  const savedTheme = localStorage.getItem("eagle-theme");
  if (savedTheme === "light" || savedTheme === "dark") initialTheme = savedTheme;
} catch {
  // Use the default theme if browser storage is unavailable.
}
setTheme(initialTheme);
setScreen();
document.querySelector("#theme-toggle").addEventListener("click", () => {
  setTheme(document.body.dataset.theme === "dark" ? "light" : "dark");
});

loadLeagues();
loadModelLeagues();
