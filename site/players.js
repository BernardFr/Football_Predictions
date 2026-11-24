// Impact players page: show one team at a time. The URL's #hash names the team, so a link opens that team.
(() => {
  const select = document.getElementById("team-select");
  const sections = document.querySelectorAll("section.team");
  if (!select || !sections.length) return;
  const show = slug => {
    if (![...sections].some(section => section.id === "team-" + slug)) slug = select.options[0].value;
    sections.forEach(section => { section.hidden = section.id !== "team-" + slug; });
    select.value = slug;
  };
  select.addEventListener("change", () => { history.replaceState(null, "", "#" + select.value); show(select.value); });
  window.addEventListener("hashchange", () => show(location.hash.slice(1)));
  show(location.hash.slice(1));
})();
