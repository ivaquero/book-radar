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
	const angleSlider = elem("angleSlider");
	const currentThetaSpan = elem("currentTheta");

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
			console.log(`参数滑块变化: ${sliderId} = ${slider.value}`);
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

	// 计算过程可视化函数
	function updateCalculationSteps(angleDeg) {
		console.log(`updateCalculationSteps: 开始, 角度=${angleDeg}°`);
		const dLambdaRatio = parseFloat(slider_dLambdaRatio.value);
		const nAnt = parseInt(slider_nAnt.value);
		const deltaT = parseFloat(slider_deltaT.value);
		const angleRad = (angleDeg * Math.PI) / 180;
		console.log(`updateCalculationSteps: 参数 - d/λ=${dLambdaRatio}, N=${nAnt}, ΔT=${deltaT}°`);

		// 物理常数
		const c0 = 299792458; // 光速 m/s
		const lambda0 = 1.0; // 假设波长为1米
		const omega0 = (2 * Math.PI * c0) / lambda0;

		// 步骤1: 计算相邻天线时间延迟 Δτ
		const deltaTau = (dLambdaRatio * lambda0 * Math.sin(angleRad)) / c0;
		elem("calc_deltaTau").innerHTML = `Δτ = (${dLambdaRatio.toFixed(3)} × ${lambda0} × sin(${angleDeg}°)) / ${c0.toExponential(2)} = ${deltaTau.toExponential(3)} s`;

		// 步骤2: 计算相位差 Δψ
		const deltaPsi = 2 * Math.PI * dLambdaRatio * Math.sin(angleRad);
		elem("calc_deltaPsi").innerHTML = `Δψ = 2π × ${dLambdaRatio.toFixed(3)} × sin(${angleDeg}°) = ${deltaPsi.toFixed(3)} rad = ${(deltaPsi * 180 / Math.PI).toFixed(1)}°`;

		// 步骤3: 计算控制相位 ΔΨ
		const deltaT_rad = (deltaT * Math.PI) / 180;
		const deltaPhi = deltaPsi - deltaT_rad;
		elem("calc_deltaPhi").innerHTML = `ΔΨ = ${deltaPsi.toFixed(3)} - ${deltaT}° = ${deltaPhi.toFixed(3)} rad = ${(deltaPhi * 180 / Math.PI).toFixed(1)}°`;

		// 步骤4: 计算主瓣方向
		let mainLobeAngleDeg = 0;
		let mainLobeStatus = "";
		const sinTheta = (deltaT * Math.PI) / 180 / (2 * Math.PI * dLambdaRatio);

		if (Math.abs(sinTheta) <= 1) {
			mainLobeAngleDeg = Math.asin(sinTheta) * (180 / Math.PI);
			mainLobeStatus = `θ<sub>MainLobe</sub> = arcsin(ω<sub>0</sub>ΔT × λ<sub>0</sub> / (2π × d<sub>Ant</sub>)) = arcsin((${deltaT}°) / (2π × ${dLambdaRatio.toFixed(3)})) = <strong style="color: #ff6670; font-size: 1.2em;">${mainLobeAngleDeg.toFixed(1)}°</strong>`;
		} else {
			mainLobeStatus = `θ<sub>MainLobe</sub> = <span style="color: #ff6670;">无解 (|sinθ| = ${Math.abs(sinTheta).toFixed(3)} > 1)</span>`;
		}
		elem("calc_mainLobeAngle").innerHTML = mainLobeStatus;

		// 步骤5: 计算阵列因子
		let afReal = 0;
		let afImag = 0;
		for (let j = 0; j < nAnt; ++j) {
			const tau = -j * deltaPhi;
			afReal += Math.cos(tau);
			afImag += Math.sin(tau);
		}
		const afMagnitude = Math.sqrt(afReal * afReal + afImag * afImag);
		const afPower = (afReal * afReal + afImag * afImag) / 4; // 相对于单天线功率
		const afPowerDb = 10 * Math.log10(afPower);

		// 增强的阵列因子显示，包含详细的dB计算过程
		elem("calc_arrayFactor").innerHTML = `
			<div style="border: 1px solid #444; padding: 8px; border-radius: 4px; background: rgba(255,255,255,0.05);">
				<div style="margin-bottom: 8px;"><strong>阵列因子计算结果：</strong></div>
				<div style="margin-bottom: 5px;">
					<span style="color: #56d4ff;">复数形式：</span>
					ArrayFactor = ${afMagnitude.toFixed(3)} ∠ ${(Math.atan2(afImag, afReal) * 180 / Math.PI).toFixed(1)}°
				</div>
				<div style="margin-bottom: 5px;">
					<span style="color: #56d4ff;">功率计算：</span>
					|ArrayFactor|² = (${afReal.toFixed(3)})² + (${afImag.toFixed(3)})² = ${(afReal * afReal + afImag * afImag).toFixed(3)}
				</div>
				<div style="margin-bottom: 5px;">
					<span style="color: #ff6670;">dB转换：</span>
					ArrayFactor<sub>dB</sub> = 10 lg(${afPower.toFixed(3)}) = <strong style="color: #ff6670; font-size: 1.1em;">${afPowerDb.toFixed(2)} dB</strong>
				</div>
				<div style="font-size: 9px; color: #888; margin-top: 5px;">
					注：功率相对于单天线归一化 (÷4)
				</div>
			</div>
		`;

		// 更新当前角度显示
		elem("currentTheta").textContent = angleDeg;

		// 计算并显示波束宽度
		console.log(`updateCalculationSteps: 调用波束宽度更新, d/λ=${dLambdaRatio}, N=${nAnt}, ΔT=${deltaT}°`);
		updateBeamWidthInfo(dLambdaRatio, nAnt, deltaT);

		// 更新相位图
		updatePhaseDiagram(deltaPhi, nAnt);

		// 更新天线阵列图
		updateArrayDiagram(deltaPhi, nAnt);

		// 更新阵列因子曲线图
		updateArrayFactorCurve(angleDeg);
	}

	// 波束宽度信息更新函数
	function updateBeamWidthInfo(dLambdaRatio, nAnt, deltaT) {
		// 调试：显示输入参数
		console.log(`波束宽度调试: d/λ=${dLambdaRatio}, 天线数=${nAnt}, ΔT=${deltaT}°`);

		// 计算半功率波束宽度
		const deltaT_rad = (deltaT * Math.PI) / 180;

		// 计算主瓣方向
		const sinThetaMain = (deltaT * Math.PI) / 180 / (2 * Math.PI * dLambdaRatio);
		let thetaMain = 0;
		let beamWidthInfo = "";

		if (Math.abs(sinThetaMain) <= 1) {
			thetaMain = Math.asin(sinThetaMain);

			// 近似计算半功率波束宽度
			const beamWidth = 50.8 / (nAnt * dLambdaRatio * Math.cos(thetaMain)); // 近似公式

			// 计算第一零点位置
			const firstNull1 = Math.asin(sinThetaMain - 1 / (nAnt * dLambdaRatio));
			const firstNull2 = Math.asin(sinThetaMain + 1 / (nAnt * dLambdaRatio));

			// 调试：显示计算结果
			console.log(`波束宽度计算: 主瓣方向=${(thetaMain * 180 / Math.PI).toFixed(2)}°, 半功率宽度=${beamWidth.toFixed(2)}°, 零点1=${(firstNull1 * 180 / Math.PI).toFixed(2)}°, 零点2=${(firstNull2 * 180 / Math.PI).toFixed(2)}°`);

			beamWidthInfo = `
				<div class="calc-step">
					<h4>波束宽度信息</h4>
					<div class="formula">半功率波束宽度 ≈ ${beamWidth.toFixed(1)}°</div>
					<div class="calc-result">主瓣方向: <strong style="color: #ff6670; font-size: 1.2em;">${(thetaMain * 180 / Math.PI).toFixed(1)}°</strong></div>
					<div class="calc-result">第一零点: ${(firstNull1 * 180 / Math.PI).toFixed(1)}°, ${(firstNull2 * 180 / Math.PI).toFixed(1)}°</div>
					<div class="calc-result">零点间距: ${((firstNull2 - firstNull1) * 180 / Math.PI).toFixed(1)}°</div>
				</div>
			`;
		} else {
			// 调试：显示无效情况
			console.log(`波束宽度计算: 主瓣方向无解, sinθ=${sinThetaMain.toFixed(3)} > 1`);

			beamWidthInfo = `
			<div class="calc-step">
				<h4>波束宽度信息</h4>
				<div class="calc-result">主瓣方向: 无解 (栅瓣条件)</div>
			</div>
		`;
		}

		// 更新或创建波束宽度信息
		let beamWidthDiv = elem("beamWidthInfo");
		if (!beamWidthDiv) {
			console.log("波束宽度调试: 创建新的波束宽度信息div");
			beamWidthDiv = document.createElement("div");
			beamWidthDiv.id = "beamWidthInfo";
			// 确保calcSteps存在
			const calcSteps = elem("calcSteps");
			if (calcSteps) {
				calcSteps.appendChild(beamWidthDiv);
				console.log("波束宽度调试: 成功添加到calcSteps");
			} else {
				console.error("波束宽度调试: 找不到calcSteps元素");
			}
		}
		if (beamWidthDiv) {
			beamWidthDiv.innerHTML = beamWidthInfo;
			console.log("波束宽度调试: 已更新波束宽度信息显示");
		}
	}

	// 阵列因子曲线图更新函数
	function updateArrayFactorCurve(currentAngle) {
		const dLambdaRatio = parseFloat(slider_dLambdaRatio.value);
		const nAnt = parseInt(slider_nAnt.value);
		const deltaT = parseFloat(slider_deltaT.value);

		// 创建或更新曲线图
		let curveSvg = elem("curveSvg");
		if (!curveSvg) {
			const curveDiv = document.createElement("div");
			curveDiv.id = "curveDiv";
			curveDiv.innerHTML = '<h4>阵列因子幅度曲线 (实线: 幅度, 虚线: dB)</h4><svg id="curveSvg" width="250" height="120"></svg><div id="curveInfo" style="font-size: 10px; margin-top: 5px;"></div>';
			elem("calcSteps").appendChild(curveDiv);
			curveSvg = elem("curveSvg");
		}

		// 清空SVG
		curveSvg.innerHTML = "";

		const width = 250;
		const height = 120;
		const margin = 15;
		const plotWidth = width - 2 * margin;
		const plotHeight = height - 2 * margin;

		// 计算阵列因子数据
		const angles = [];
		const afValues = [];
		const afDbValues = [];
		const deltaT_rd = (deltaT * Math.PI) / 180;

		for (let angle = -90; angle <= 90; angle += 2) {
			const angleRad = (angle * Math.PI) / 180;
			let afReal = 0;
			let afImag = 0;

			for (let j = 0; j < nAnt; ++j) {
				const tau = -j * (2 * Math.PI * dLambdaRatio * Math.sin(angleRad) - deltaT_rd);
				afReal += Math.cos(tau);
				afImag += Math.sin(tau);
			}

			const afMagnitude = Math.sqrt(afReal * afReal + afImag * afImag) / nAnt;
			const afPower = afMagnitude * afMagnitude;
			const afDb = afPower > 0 ? 10 * Math.log10(afPower) : -40; // 避免log(0)
			angles.push(angle);
			afValues.push(afMagnitude);
			afDbValues.push(afDb);
		}

		// 找到最大值和最小值
		const maxAf = Math.max(...afValues);
		const minAf = Math.min(...afValues);
		const maxDb = Math.max(...afDbValues);
		const minDb = Math.min(...afDbValues);

		// 绘制坐标轴
		const xAxis = document.createElementNS("http://www.w3.org/2000/svg", "line");
		xAxis.setAttribute("x1", margin);
		xAxis.setAttribute("y1", height - margin);
		xAxis.setAttribute("x2", width - margin);
		xAxis.setAttribute("y2", height - margin);
		xAxis.setAttribute("stroke", "#666");
		xAxis.setAttribute("stroke-width", "1");
		curveSvg.appendChild(xAxis);

		const yAxis = document.createElementNS("http://www.w3.org/2000/svg", "line");
		yAxis.setAttribute("x1", margin);
		yAxis.setAttribute("y1", margin);
		yAxis.setAttribute("x2", margin);
		yAxis.setAttribute("y2", height - margin);
		yAxis.setAttribute("stroke", "#666");
		yAxis.setAttribute("stroke-width", "1");
		curveSvg.appendChild(yAxis);

		// 绘制阵列因子幅度曲线
		let pathData = "";
		for (let i = 0; i < angles.length; i++) {
			const x = margin + (angles[i] + 90) * plotWidth / 180;
			const y = height - margin - (afValues[i] - minAf) * plotHeight / (maxAf - minAf);
			pathData += (i === 0 ? "M" : "L") + x + "," + y;
		}

		const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
		path.setAttribute("d", pathData);
		path.setAttribute("fill", "none");
		path.setAttribute("stroke", "#56d4ff");
		path.setAttribute("stroke-width", "2");
		curveSvg.appendChild(path);

		// 绘制阵列因子dB曲线（使用虚线）
		let dbPathData = "";
		for (let i = 0; i < angles.length; i++) {
			const x = margin + (angles[i] + 90) * plotWidth / 180;
			const y = height - margin - (afDbValues[i] - minDb) * plotHeight / (maxDb - minDb);
			dbPathData += (i === 0 ? "M" : "L") + x + "," + y;
		}

		const dbPath = document.createElementNS("http://www.w3.org/2000/svg", "path");
		dbPath.setAttribute("d", dbPathData);
		dbPath.setAttribute("fill", "none");
		dbPath.setAttribute("stroke", "#ff6670");
		dbPath.setAttribute("stroke-width", "1.5");
		dbPath.setAttribute("stroke-dasharray", "3,2");
		curveSvg.appendChild(dbPath);

		// 标记当前角度位置
		const currentX = margin + (currentAngle + 90) * plotWidth / 180;
		const currentIndex = Math.round((currentAngle + 90) / 2);
		const currentY = height - margin - (afValues[currentIndex] - minAf) * plotHeight / (maxAf - minAf);
		const currentDbY = height - margin - (afDbValues[currentIndex] - minDb) * plotHeight / (maxDb - minDb);

		// 幅度标记（蓝色）
		const marker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
		marker.setAttribute("cx", currentX);
		marker.setAttribute("cy", currentY);
		marker.setAttribute("r", "3");
		marker.setAttribute("fill", "#56d4ff");
		curveSvg.appendChild(marker);

		// dB标记（红色，稍微偏移）
		const dbMarker = document.createElementNS("http://www.w3.org/2000/svg", "circle");
		dbMarker.setAttribute("cx", currentX);
		dbMarker.setAttribute("cy", currentDbY);
		dbMarker.setAttribute("r", "2");
		dbMarker.setAttribute("fill", "#ff6670");
		curveSvg.appendChild(dbMarker);

		// 添加dB值信息到页面
		const curveInfo = elem("curveInfo");
		if (curveInfo) {
			curveInfo.innerHTML = `当前角度 ${currentAngle}°: 幅度 = ${afValues[currentIndex].toFixed(3)}, dB = ${afDbValues[currentIndex].toFixed(1)} dB`;
		}
	}

	// 相位图更新函数
	function updatePhaseDiagram(deltaPhi, nAnt) {
		// 创建或更新相位图
		let phaseSvg = elem("phaseSvg");
		if (!phaseSvg) {
			// 创建相位图容器
			const phaseDiv = document.createElement("div");
			phaseDiv.id = "phaseDiv";
			phaseDiv.innerHTML = '<h4>天线相位关系图</h4><svg id="phaseSvg" width="250" height="180"></svg>';
			elem("calcSteps").appendChild(phaseDiv);
			phaseSvg = elem("phaseSvg");
		}

		// 清空SVG
		phaseSvg.innerHTML = "";

		const centerX = 125;
		const centerY = 90;
		const radius = 60;

		// 绘制坐标轴
		for (let i = 0; i < nAnt; i++) {
			const phase = i * deltaPhi;
			const x = centerX + radius * Math.cos(phase);
			const y = centerY + radius * Math.sin(phase);

			// 绘制相位向量
			const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
			line.setAttribute("x1", centerX);
			line.setAttribute("y1", centerY);
			line.setAttribute("x2", x);
			line.setAttribute("y2", y);
			line.setAttribute("stroke", "#56d4ff");
			line.setAttribute("stroke-width", "2");
			phaseSvg.appendChild(line);

			// 绘制天线编号
			const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
			text.setAttribute("x", x + 8);
			text.setAttribute("y", y + 4);
			text.setAttribute("fill", "#ffe970");
			text.setAttribute("font-size", "10");
			text.textContent = `A${i}`;
			phaseSvg.appendChild(text);
		}

		// 绘制参考圆
		const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
		circle.setAttribute("cx", centerX);
		circle.setAttribute("cy", centerY);
		circle.setAttribute("r", radius);
		circle.setAttribute("fill", "none");
		circle.setAttribute("stroke", "#666");
		circle.setAttribute("stroke-dasharray", "5,5");
		phaseSvg.appendChild(circle);
	}

	// 天线阵列图更新函数
	function updateArrayDiagram(deltaPhi, nAnt) {
		// 创建或更新阵列图
		let arraySvg = elem("arraySvg");
		if (!arraySvg) {
			// 创建阵列图容器
			const arrayDiv = document.createElement("div");
			arrayDiv.id = "arrayDiv";
			arrayDiv.innerHTML = '<h4>天线阵列几何图</h4><svg id="arraySvg" width="250" height="120"></svg>';
			elem("calcSteps").appendChild(arrayDiv);
			arraySvg = elem("arraySvg");
		}

		// 清空SVG
		arraySvg.innerHTML = "";

		const startX = 40;
		const spacing = 170 / Math.max(nAnt - 1, 1);
		const centerY = 60;

		// 绘制天线阵列
		for (let i = 0; i < nAnt; i++) {
			const x = startX + i * spacing;

			// 绘制天线元素
			const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
			circle.setAttribute("cx", x);
			circle.setAttribute("cy", centerY);
			circle.setAttribute("r", "6");
			circle.setAttribute("fill", "#56d4ff");
			circle.setAttribute("stroke", "#ffe970");
			circle.setAttribute("stroke-width", "1.5");
			arraySvg.appendChild(circle);

			// 绘制相位指示器
			const phase = i * deltaPhi;
			const phaseText = document.createElementNS("http://www.w3.org/2000/svg", "text");
			phaseText.setAttribute("x", x);
			phaseText.setAttribute("y", centerY - 15);
			phaseText.setAttribute("fill", "#ff6670");
			phaseText.setAttribute("font-size", "8");
			phaseText.setAttribute("text-anchor", "middle");
			phaseText.textContent = `${(phase * 180 / Math.PI).toFixed(0)}°`;
			arraySvg.appendChild(phaseText);

			// 绘制天线编号
			const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
			label.setAttribute("x", x);
			label.setAttribute("y", centerY + 20);
			label.setAttribute("fill", "#ffe970");
			label.setAttribute("font-size", "10");
			label.setAttribute("text-anchor", "middle");
			label.textContent = `A${i}`;
			arraySvg.appendChild(label);
		}
	}

	// 设置角度滑块事件
	angleSlider.oninput = function () {
		const angle = parseInt(this.value);
		console.log(`角度滑块变化: ${angle}°`);
		updateCalculationSteps(angle);
		updateBeamIndicator(angle);
		updatePlot(); // 添加这一行来更新波束宽度信息
	};

	// 参数滑块变化时也更新计算过程
	const originalUpdatePlot = updatePlot;
	updatePlot = function () {
		console.log("updatePlot: 开始更新");
		originalUpdatePlot();
		const currentAngle = parseInt(angleSlider.value);
		console.log(`updatePlot: 当前角度=${currentAngle}°`);
		updateCalculationSteps(currentAngle);
		updateBeamIndicator(currentAngle);

		// 调试：显示当前参数和主瓣方向
		const dLambdaRatio = parseFloat(slider_dLambdaRatio.value);
		const deltaT = parseFloat(slider_deltaT.value);
		const sinTheta = (deltaT * Math.PI) / 180 / (2 * Math.PI * dLambdaRatio);
		if (Math.abs(sinTheta) <= 1) {
			const thetaMain = Math.asin(sinTheta) * (180 / Math.PI);
			console.log(`调试: deltaT=${deltaT}°, d/λ=${dLambdaRatio}, 主瓣方向=${thetaMain.toFixed(2)}°`);
		} else {
			console.log(`调试: deltaT=${deltaT}°, d/λ=${dLambdaRatio}, 主瓣方向=无解`);
		}
	};

	// 波束指向指示器
	function updateBeamIndicator(angleDeg) {
		// 在极坐标图上添加当前角度指示器
		const existingIndicator = elem("beamIndicator");
		if (existingIndicator) {
			existingIndicator.remove();
		}

		// 获取SVG元素
		const svg = elem("display");
		if (!svg) return;

		// 创建指示器线
		const angleRad = (angleDeg * Math.PI) / 180;
		const centerX = 300; // 假设中心在300,300
		const centerY = 300;
		const radius = 200;

		const endX = centerX + radius * Math.cos(angleRad - Math.PI / 2); // -90度调整因为SVG坐标系
		const endY = centerY + radius * Math.sin(angleRad - Math.PI / 2);

		const indicator = document.createElementNS("http://www.w3.org/2000/svg", "line");
		indicator.id = "beamIndicator";
		indicator.setAttribute("x1", centerX);
		indicator.setAttribute("y1", centerY);
		indicator.setAttribute("x2", endX);
		indicator.setAttribute("y2", endY);
		indicator.setAttribute("stroke", "#ff6670");
		indicator.setAttribute("stroke-width", "3");
		indicator.setAttribute("stroke-dasharray", "5,5");
		indicator.setAttribute("opacity", "0.8");

		svg.appendChild(indicator);
	}

	updatePlot();
	autoScalling();

	// 初始化计算步骤显示
	updateCalculationSteps(0);
	updateBeamIndicator(0);
};
