export function openPanel(element: string | null | HTMLElement) {
  let el: HTMLElement | null = null;
  if (typeof element === "string") {
    el = document.getElementById(element);
  } else {
    el = element;
  }
  if (!el) return;
  el.classList.remove("closed");
  el.classList.add("opened");
}
