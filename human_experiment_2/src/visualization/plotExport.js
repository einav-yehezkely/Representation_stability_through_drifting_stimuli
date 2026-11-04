// Export the researcher PCA plot as SVG or PNG (downloaded by the browser).

function triggerDownload(fileName, blob) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

export function serializeSvg(svg) {
  const clone = svg.cloneNode(true);
  clone.querySelectorAll('[display="none"]').forEach((node) => node.remove());
  const [, , w, hgt] = clone.getAttribute("viewBox").split(" ").map(Number);
  clone.setAttribute("width", w);
  clone.setAttribute("height", hgt);
  return `<?xml version="1.0" encoding="UTF-8"?>\n${new XMLSerializer().serializeToString(clone)}`;
}

export function exportSvg(svg, fileName) {
  triggerDownload(fileName, new Blob([serializeSvg(svg)], { type: "image/svg+xml" }));
}

export async function exportPng(svg, fileName, scale = 2) {
  const text = serializeSvg(svg);
  const [, , w, hgt] = svg.getAttribute("viewBox").split(" ").map(Number);
  const url = URL.createObjectURL(new Blob([text], { type: "image/svg+xml" }));
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const canvas = document.createElement("canvas");
    canvas.width = w * scale;
    canvas.height = hgt * scale;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "white";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
    triggerDownload(fileName, blob);
  } finally {
    URL.revokeObjectURL(url);
  }
}
