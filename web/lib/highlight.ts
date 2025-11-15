/**
 * Highlights one or more DOM elements by their id(s), ensuring only those are highlighted.
 * Removes the "highlight" class from all previously highlighted elements.
 * When clicking a highlighted element, remove its highlight.
 * @param ids - A single id string or an array of id strings.
 */
export function highlightId(ids: string | string[]) {
  const idList = Array.isArray(ids) ? ids : [ids];
  console.log("highlighting", idList);

  // Remove previous highlights
  document.querySelectorAll(".highlight").forEach((el) => {
    el.classList.remove("highlight");
    el.removeEventListener("click", removeHighlightHandler);
  });

  // Add highlight and event handler to current IDs
  idList.forEach((id) => {
    const el = document.getElementById(id);
    if (el) {
      el.classList.add("highlight");
      el.addEventListener("click", removeHighlightHandler);
    }
  });
}

// Simple reusable handler
function removeHighlightHandler(this: HTMLElement) {
  this.classList.remove("highlight");
  this.removeEventListener("click", removeHighlightHandler);
}
