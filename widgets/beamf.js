function elem(id) {
	return document.getElementById(id);
}

function showValue(objName, value, maxValue) {
	const obj = elem(objName);
	obj.innerHTML = `${value} / ${maxValue}`;
}

function arrayFactor(dLambdaRatio, nAnt, deltaT) {
	const data = new Array();
	const deltaT_rd = (deltaT * Math.PI) / 180;
	for (let i = 0; i < 360; ++i) {
		data.push(i); // Theta
		let afReal = 0;
		let afImag = 0;
		const angle_rd = (i * Math.PI) / 180;
		for (let j = 0; j < nAnt; ++j) {
			const tau =
				-j * (2 * Math.PI * dLambdaRatio * Math.sin(angle_rd) - deltaT_rd);
			afReal += Math.cos(tau);
			afImag += Math.sin(tau);
		}
		const afPowerDb = 10 * Math.log10((afReal * afReal + afImag * afImag) / 4);
		data.push(afPowerDb);
	}
	data.push(data[0]);
	data.push(data[1]); // Loop back
	return data;
}

function createSvg(w = 600, h = 600, svgId = "display") {
	const svg = new Object();

	svg.init = function () {
		// this.msvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
		// document.body.appendChild(this.msvg);
		this.msvg = elem(svgId);
		this.msvg.innerHTML = ""; // Clear svg if exist
		this.msvg.setAttribute("width", String(w));
		this.msvg.setAttribute("height", String(h));
		this.msvg.setAttribute("border", "2px solid rgb(24, 90, 167)");

		this.polarScattersArray = new Array();
		this.hasPolarScatters = false;
		this.polarW = 0;
		this.polarH = 0;
		this.polarCx = 0;
		this.polarCy = 0;
		this.polarMin = 0;
		this.polarMax = 0;
		this.polarPointStr = "";

		this.polarAxisArray = new Array();
		this.hasPolarAxis = false;

		this.yellowColor = "#ffe970";
		this.blueColor = "#56d4ff";
		this.redColor = "#ff6670";
	};

	svg.init();

	svg.setPolarAxis = function (r, cx, cy, minlim, maxlim) {
		this.polarR = r;
		this.polarCx = cx;
		this.polarCy = cy;
		this.polarMin = minlim;
		this.polarMax = maxlim;

		if (this.hasPolarAxis) {
			for (let i = 0; i < this.polarAxisArray.length; ++i) {
				this.msvg.removeChild(this.polarAxisArray[i]);
			}
		}
		this.polarAxisArray = new Array();

		const polarBorder = document.createElementNS(
			"http://www.w3.org/2000/svg",
			"circle",
		);
		polarBorder.setAttribute("cx", String(this.polarCx));
		polarBorder.setAttribute("cy", String(this.polarCy));
		polarBorder.setAttribute("r", String(this.polarR));
		polarBorder.setAttribute("fill", "black");
		polarBorder.setAttribute(
			"style",
			`stroke:${this.yellowColor};stroke-width:2;stroke-opacity:1.0`,
		);
		this.polarAxisArray.push(polarBorder);
		this.msvg.appendChild(polarBorder);

		const range = this.polarMax - this.polarMin;
		const section = 10 ** (Math.ceil(Math.log10(range)) - 1);

		const sectionStart = Math.abs(this.polarMin % section) + this.polarMin;
		const nSection = Math.floor((this.polarMax - sectionStart) / section) + 1;

		const fontSize = 16;

		for (let i = 0; i < nSection; ++i) {
			const grid = document.createElementNS(
				"http://www.w3.org/2000/svg",
				"circle",
			);
			grid.setAttribute("cx", String(this.polarCx));
			grid.setAttribute("cy", String(this.polarCy));
			const rLocation =
				(this.polarR * (sectionStart + i * section - this.polarMin)) / range;
			grid.setAttribute("r", String(rLocation));
			grid.setAttribute("fill", "none");
			grid.setAttribute(
				"style",
				"stroke:white;stroke-width:0.5;stroke-opacity:0.5",
			);
			this.polarAxisArray.push(grid);
			this.msvg.appendChild(grid);

			const text = document.createElementNS(
				"http://www.w3.org/2000/svg",
				"text",
			);
			text.innerHTML = String(sectionStart + i * section);
			text.setAttribute("x", String(this.polarCx + rLocation));
			text.setAttribute("y", String(this.polarCy + fontSize));
			text.setAttribute("fill", "white");
			text.setAttribute("style", "fill-opacity:0.5");
			text.setAttribute("font-size", `${fontSize}px`);
			this.polarAxisArray.push(text);
			this.msvg.appendChild(text);
		}

		const angleSection = 30;
		const nAngleSction = 360 / angleSection;
		const textR = this.polarR + fontSize * 1.5;
		for (let i = 0; i < nAngleSction; ++i) {
			let angle = angleSection * i;
			const grid = document.createElementNS(
				"http://www.w3.org/2000/svg",
				"line",
			);
			grid.setAttribute("x1", String(this.polarCx));
			grid.setAttribute("y1", String(this.polarCy));
			grid.setAttribute(
				"x2",
				String(this.polarCx + this.polarR * Math.cos((angle / 180) * Math.PI)),
			);
			grid.setAttribute(
				"y2",
				String(this.polarCy - this.polarR * Math.sin((angle / 180) * Math.PI)),
			);
			grid.setAttribute("fill", "none");
			grid.setAttribute(
				"style",
				"stroke:white;stroke-width:0.5;stroke-opacity:0.5",
			);
			this.polarAxisArray.push(grid);
			this.msvg.appendChild(grid);

			const text = document.createElementNS(
				"http://www.w3.org/2000/svg",
				"text",
			);
			if (angle > 180) {
				angle -= 360;
			}
			text.innerHTML = String(angle);
			if (angle === 0) {
				text.innerHTML = `${text.innerHTML}°`;
			}

			text.setAttribute(
				"x",
				String(
					this.polarCx + textR * Math.cos((angle / 180) * Math.PI) - fontSize,
				),
			);
			text.setAttribute(
				"y",
				String(
					this.polarCx -
						textR * Math.sin((angle / 180) * Math.PI) +
						fontSize / 4,
				),
			);
			text.setAttribute("fill", this.yellowColor);
			text.setAttribute("style", "fill-opacity:1.0");
			text.setAttribute("font-size", `${fontSize}px`);
			this.polarAxisArray.push(text);
			this.msvg.appendChild(text);
		}
		this.hasPolarAxis = true;
	};

	svg.drawPolarPoints = function (dLambdaRatio, nAnt, deltaT) {
		const dataPoints = arrayFactor(dLambdaRatio, nAnt, deltaT);
		if (!this.hasPolarAxis) {
			this.setPolarAxis();
		}
		this.polarScatters = document.createElementNS(
			"http://www.w3.org/2000/svg",
			"polyline",
		);
		this.polarPointStr = "";
		let pha = 0;
		for (let i = 0; i < dataPoints.length; ++i) {
			if (i % 2 === 0) {
				// phase value
				pha = (dataPoints[i] / 180) * Math.PI;
			} else {
				// amplitude value, process it
				let amp = dataPoints[i];
				if (amp > this.polarMax) {
					amp = this.polarMax;
				} else if (amp < this.polarMin) {
					amp = this.polarMin;
				}
				const ratioLen =
					((amp - this.polarMin) * this.polarR) /
					(this.polarMax - this.polarMin);
				const _x = ratioLen * Math.cos(pha) + this.polarCx;
				const _y = -ratioLen * Math.sin(pha) + this.polarCy;
				this.polarPointStr += `${_x},${_y}`;
				if (i !== dataPoints.length - 1) {
					this.polarPointStr += " ";
				}
			}
		}
		const polarPoints = document.createElementNS(
			"http://www.w3.org/2000/svg",
			"polyline",
		);
		polarPoints.setAttribute("points", this.polarPointStr);
		polarPoints.setAttribute("stroke", this.yellowColor);
		polarPoints.setAttribute("stroke-width", "2");
		polarPoints.setAttribute("fill", this.yellowColor);
		polarPoints.setAttribute("style", "fill-opacity:0.3");
		this.msvg.appendChild(polarPoints);
		this.polarScattersArray.push(polarPoints);
		this.hasPolarScatters = true;
	};

	svg.drawArc = function (
		phaseDg,
		styleStr = "stroke:white;stroke-width:4.0;stroke-opacity:1.0",
	) {
		// SVG:  M x0 y0 A rx ry rotation large-arc-flag sweep-flag x1 y1
		const x0 = this.polarCx + this.polarR;
		const y0 = this.polarCy;
		const rx = this.polarR;
		const ry = this.polarR;
		const rotation = 0;
		const large_arc_flag = 0;
		let sweep_flag = 0;
		if (phaseDg < 0) {
			sweep_flag = 1;
		}
		const x1 = this.polarCx + this.polarR * Math.cos((phaseDg * Math.PI) / 180);
		const y1 = this.polarCy - this.polarR * Math.sin((phaseDg * Math.PI) / 180);
		const arc = document.createElementNS("http://www.w3.org/2000/svg", "path");
		let pathStr = `M ${x0} ${y0} A `;
		pathStr += [rx, ry, rotation, large_arc_flag, sweep_flag, x1, y1].join(" ");
		arc.setAttribute("d", pathStr);
		arc.setAttribute("fill", "none");
		arc.setAttribute("style", styleStr);
		this.polarScattersArray.push(arc);
		this.msvg.appendChild(arc);
	};

	svg.drawAngleLine = function (
		phaseDg,
		styleStr = "stroke:white;stroke-width:2.0;stroke-opacity:1.0",
	) {
		const grid = document.createElementNS("http://www.w3.org/2000/svg", "line");
		grid.setAttribute("x1", String(this.polarCx));
		grid.setAttribute("y1", String(this.polarCy));
		grid.setAttribute(
			"x2",
			String(this.polarCx + this.polarR * Math.cos((phaseDg / 180) * Math.PI)),
		);
		grid.setAttribute(
			"y2",
			String(this.polarCy - this.polarR * Math.sin((phaseDg / 180) * Math.PI)),
		);
		grid.setAttribute("fill", "none");
		grid.setAttribute("style", styleStr);
		this.polarScattersArray.push(grid);
		this.msvg.appendChild(grid);
	};

	svg.drawDeltaTArc = function (phaseDg) {
		this.drawArc(phaseDg, "stroke:white;stroke-width:4.0;stroke-opacity:1.0");
	};

	svg.drawDeltaT = function (phaseDg) {
		this.drawAngleLine(
			phaseDg,
			"stroke:white;stroke-width:2.0;stroke-opacity:1.0",
		);
	};

	svg.drawMainLobeArc = function (phaseDg, overflow = false) {
		if (overflow) {
			this.drawArc(
				phaseDg,
				`stroke:${this.redColor};stroke-width:4.0;stroke-opacity:1.0`,
			);
		} else {
			this.drawArc(
				phaseDg,
				`stroke:${this.blueColor};stroke-width:4.0;stroke-opacity:1.0`,
			);
		}
	};

	svg.drawMainLobe = function (phaseDg, overflow = false) {
		if (overflow) {
			this.drawAngleLine(
				phaseDg,
				`stroke:${this.redColor};stroke-width:2.0;stroke-opacity:1.0`,
			);
			this.drawAngleLine(
				phaseDg - 180,
				`stroke:${this.redColor};stroke-width:1.0;stroke-opacity:0.5`,
			);
		} else {
			this.drawAngleLine(
				phaseDg,
				`stroke:${this.blueColor};stroke-width:2.0;stroke-opacity:1.0`,
			);
			this.drawAngleLine(
				phaseDg - 180,
				`stroke:${this.blueColor};stroke-width:1.0;stroke-opacity:0.5`,
			);
		}
	};

	svg.drawAntPoint = function (cx, cy) {
		const circle = document.createElementNS(
			"http://www.w3.org/2000/svg",
			"circle",
		);
		circle.setAttribute("cx", String(cx));
		circle.setAttribute("cy", String(cy));
		circle.setAttribute("r", "3");
		circle.setAttribute("fill", this.blueColor);
		circle.setAttribute(
			"style",
			"fill-opacity:0.9;stroke:white;stroke-width:2.0;stroke-opacity:1.0",
		);
		this.polarScattersArray.push(circle);
		this.msvg.appendChild(circle);
	};

	svg.drawNAnt = function (dLambdaRatio, nAnt, max_dLambdaRatio, max_nAnt) {
		const minY = this.polarCy - this.polarR;
		const maxY = this.polarCy + this.polarR;
		const rangeY = maxY - minY;
		const antGap =
			(rangeY / (max_nAnt - 1)) * Math.sqrt(dLambdaRatio / max_dLambdaRatio);
		const sumGap = antGap * (nAnt - 1);
		const startY = minY + (rangeY - sumGap) / 2;
		for (let i = 0; i < nAnt; ++i) {
			this.drawAntPoint(this.polarCx, startY + antGap * i);
		}
	};

	svg.drawCover = function () {
		// SVG:  M x0 y0 A rx ry rotation large-arc-flag sweep-flag x1 y1
		const coverR = this.polarR + 2;
		const x0 = this.polarCx;
		const y0 = this.polarCy - coverR;
		const rx = coverR;
		const ry = coverR;
		const rotation = 0;
		const large_arc_flag = 0;
		const sweep_flag = 0;
		const x1 = this.polarCx;
		const y1 = this.polarCy + coverR;
		const arc = document.createElementNS("http://www.w3.org/2000/svg", "path");
		let pathStr = `M ${x0} ${y0} A `;
		pathStr += [rx, ry, rotation, large_arc_flag, sweep_flag, x1, y1].join(" ");
		arc.setAttribute("d", pathStr);
		arc.setAttribute("fill", "black");
		arc.setAttribute("style", "fill-opacity:0.5;");
		this.polarScattersArray.push(arc);
		this.msvg.appendChild(arc);
	};

	svg.clearPolarPoints = function () {
		if (this.hasPolarScatters) {
			for (let i = 0; i < this.polarScattersArray.length; ++i) {
				this.msvg.removeChild(this.polarScattersArray[i]);
			}
			this.polarScattersArray = new Array();
			this.hasPolarScatters = false;
		}
	};
	return svg;
}

window.onload = () => {
	const mySvg = createSvg(600, 600);
	mySvg.setPolarAxis(250, 300, 300, -40, 25);

	// default values
	const default_dLambdaRatio = 0.3;
	const default_nAnt = 8;
	const default_deltaT = 0;

	let dLambdaRatio = default_dLambdaRatio;
	let nAnt = default_nAnt;
	let deltaT = default_deltaT;

	const max_dLambdaRatio = 1.2;
	const max_nAnt = 16;

	let show_deltaT = true;
	let show_nAnt = true;

	const slider_dLambdaRatio = elem("slider_dLambdaRatio");
	const slider_nAnt = elem("slider_nAnt");
	const slider_deltaT = elem("slider_deltaT");

	const btn_hide_deltaT = elem("hide_deltaT_btn");
	const btn_hide_nAnt = elem("hide_nAnt_btn");
	const btn_reset_deltaT = elem("reset_deltaT_btn");
	const btn_reset_all = elem("reset_all_btn");

	slider_dLambdaRatio.value = dLambdaRatio;
	slider_nAnt.value = nAnt;
	slider_deltaT.value = deltaT;

	slider_dLambdaRatio.max = max_dLambdaRatio;
	slider_nAnt.max = max_nAnt;

	function showAllValues() {
		showValue(
			"value_dLambdaRatio",
			slider_dLambdaRatio.value,
			slider_dLambdaRatio.max,
		);
		showValue("value_nAnt", slider_nAnt.value, slider_nAnt.max);
		showValue("value_deltaT", slider_deltaT.value, slider_deltaT.max);
	}

	function updatePlot() {
		dLambdaRatio = slider_dLambdaRatio.value;
		nAnt = slider_nAnt.value;
		deltaT = slider_deltaT.value;
		mySvg.clearPolarPoints();

		let sinTheta = (deltaT * Math.PI) / 180 / (2 * Math.PI * dLambdaRatio);
		let thetaOverflow = false;
		if (sinTheta > 1) {
			sinTheta = 1;
			thetaOverflow = true;
		} else if (sinTheta < -1) {
			sinTheta = -1;
			thetaOverflow = true;
		} else {
			thetaOverflow = false;
		}
		const mainLobePhase = Math.asin(sinTheta) * (180 / Math.PI);

		if (show_deltaT) {
			mySvg.drawDeltaT(deltaT);
			mySvg.drawMainLobe(mainLobePhase, thetaOverflow);
		}
		mySvg.drawPolarPoints(dLambdaRatio, nAnt, deltaT);
		if (show_deltaT) {
			mySvg.drawDeltaTArc(deltaT);
			mySvg.drawMainLobeArc(mainLobePhase, thetaOverflow);
		}
		mySvg.drawCover();
		if (show_nAnt) {
			mySvg.drawNAnt(dLambdaRatio, nAnt, max_dLambdaRatio, max_nAnt);
		}
		showAllValues();
	}

	function showToolTip(tipId, sliderId, unit = "") {
		elem(tipId).style.visibility = "visible";
		elem(tipId).textContent = elem(sliderId).value + unit;
		const rangeRatio =
			(91 * (elem(sliderId).value - elem(sliderId).min)) /
			(elem(sliderId).max - elem(sliderId).min);
		elem(tipId).style.left = `calc(${rangeRatio}% - 35px)`;
	}

	function setupSlider(tipId, sliderId, unit = "") {
		const slider = elem(sliderId);
		slider.hasTimeOut = false;
		slider.timer = null;
		slider.hideTip = () => {
			elem(tipId).style.visibility = "hidden";
			slider.hasTimeOut = false;
		};
		slider.onmouseenter = () => {
			showToolTip(tipId, sliderId, unit);
			if (slider.hasTimeOut) {
				clearTimeout(slider.timer);
			}
			slider.timer = setTimeout(slider.hideTip, 600);
			slider.hasTimeOut = true;
		};
		slider.oninput = () => {
			showToolTip(tipId, sliderId, unit);
			updatePlot();
			if (slider.hasTimeOut) {
				clearTimeout(slider.timer);
			}
			slider.timer = setTimeout(slider.hideTip, 600);
			slider.hasTimeOut = true;
		};
		slider.onmouseout = () => {
			if (slider.hasTimeOut) {
				clearTimeout(slider.timer);
			}
			slider.hideTip();
			slider.hasTimeOut = false;
		};
	}
	setupSlider("tip_dLambdaRatio", "slider_dLambdaRatio");
	setupSlider("tip_nAnt", "slider_nAnt");
	setupSlider("tip_deltaT", "slider_deltaT", "°");

	btn_hide_deltaT.onclick = () => {
		if (show_deltaT) {
			btn_hide_deltaT.innerHTML = "显示 ΔT";
		} else {
			btn_hide_deltaT.innerHTML = "隐藏 ΔT";
		}
		show_deltaT = !show_deltaT;
		updatePlot();
	};

	btn_hide_nAnt.onclick = () => {
		if (show_nAnt) {
			btn_hide_nAnt.innerHTML = "显示 N<sub>Ant</sub>";
		} else {
			btn_hide_nAnt.innerHTML = "隐藏 N<sub>Ant</sub>";
		}
		show_nAnt = !show_nAnt;
		updatePlot();
	};

	btn_reset_deltaT.onclick = () => {
		slider_deltaT.value = default_deltaT;
		updatePlot();
	};

	btn_reset_all.onclick = () => {
		slider_dLambdaRatio.value = default_dLambdaRatio;
		slider_nAnt.value = default_nAnt;
		slider_deltaT.value = default_deltaT;
		updatePlot();
	};

	function autoScalling() {
		let winWidth = 800;
		let winHeight = 600;
		if (window.innerWidth) winWidth = window.innerWidth;
		else if (document.body?.clientWidth) winWidth = document.body.clientWidth;
		if (window.innerHeight) winHeight = window.innerHeight;
		else if (document.body?.clientHeight)
			winHeight = document.body.clientHeight;
		if (
			document.documentElement?.clientHeight &&
			document.documentElement.clientWidth
		) {
			winHeight = document.documentElement.clientHeight;
			winWidth = document.documentElement.clientWidth;
		}
		let ratio = 1;
		if (winHeight / winWidth > 1.2) {
			// Maybe a cell phone?
			ratio = winWidth / 610;
			// console.log("cell" + ratio);
		} else {
			// Maybe a PC?
			ratio = winHeight / 800;
			// console.log("pc" + ratio);
		}
		if (ratio < 1) {
			elem("boxViewWrap").setAttribute(
				"style",
				`height: ${elem("boxViewWrap").clientHeight * ratio}px`,
			);
		}
		elem("boxViewWrap").style.transformOrigin = "50% 0";
		elem("boxViewWrap").style.transform = `scale(${ratio})`;
	}

	updatePlot();
	autoScalling();
};
