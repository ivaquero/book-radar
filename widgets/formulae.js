document.addEventListener("DOMContentLoaded", () => {
	renderMathInElement(document.body, {
		delimiters: [{ left: "$", right: "$", display: true }],
		throwOnError: false,
	});
});
