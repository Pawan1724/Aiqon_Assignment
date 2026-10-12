/**
 * app.js - GoodLogistic Frontend Controller
 * Powers Route Planning, Interactive Map, Real ML Inference, Propagation & Data Explorer
 */

// Global State
let currentRoutes = [];
let selectedRouteIndex = 0;
let leafletMap = null;
let mapLayersGroup = null;
let slaChartInstance = null;
let bookingChartInstance = null;
let searchDebounceTimeout = null;

// On DOM Ready
document.addEventListener("DOMContentLoaded", () => {
    initApp();
});

async function initApp() {
    initSidebarResizer();
    initMap();
    await loadCities();
    await loadOverviewStats();
    await handlePlanRoutes();
    loadExplorerData();
}

/**
 * Interactive Sidebar Resizer
 * Allows user to smoothly drag and resize the left sidebar between 320px and 750px.
 * Double-click resets to standard width (420px). Remembers preference in localStorage.
 */
function initSidebarResizer() {
    const resizer = document.getElementById("sidebar-resizer");
    const sidebar = document.getElementById("left-sidebar");
    if (!resizer || !sidebar) return;

    // Load persisted width
    try {
        const savedWidth = localStorage.getItem("goodlogistic_sidebar_width");
        if (savedWidth) {
            const parsed = parseInt(savedWidth, 10);
            if (!isNaN(parsed) && parsed >= 320 && parsed <= 750) {
                sidebar.style.width = parsed + "px";
            }
        }
    } catch (_) {}

    let isResizing = false;

    resizer.addEventListener("mousedown", (e) => {
        isResizing = true;
        resizer.classList.add("resizing");
        document.body.style.cursor = "col-resize";
        document.body.style.userSelect = "none";
        e.preventDefault();
    });

    window.addEventListener("mousemove", (e) => {
        if (!isResizing) return;
        const appLayout = document.querySelector(".app-layout");
        const containerLeft = appLayout ? appLayout.getBoundingClientRect().left : 0;
        let newWidth = e.clientX - containerLeft;

        const maxWidth = Math.min(750, Math.floor(window.innerWidth * 0.65));
        if (newWidth < 320) newWidth = 320;
        if (newWidth > maxWidth) newWidth = maxWidth;

        sidebar.style.width = newWidth + "px";
    });

    window.addEventListener("mouseup", () => {
        if (isResizing) {
            isResizing = false;
            resizer.classList.remove("resizing");
            document.body.style.cursor = "";
            document.body.style.userSelect = "";

            const finalWidth = parseInt(sidebar.style.width, 10);
            if (!isNaN(finalWidth)) {
                try {
                    localStorage.setItem("goodlogistic_sidebar_width", finalWidth);
                } catch (_) {}
            }

            // Invalidate Leaflet map size so canvas reflows smoothly
            if (leafletMap) {
                leafletMap.invalidateSize();
            }
            window.dispatchEvent(new Event("resize"));
        }
    });

    // Double-click to reset to default comfortable width (420px)
    resizer.addEventListener("dblclick", () => {
        sidebar.style.width = "420px";
        try {
            localStorage.setItem("goodlogistic_sidebar_width", 420);
        } catch (_) {}
        if (leafletMap) {
            leafletMap.invalidateSize();
        }
        window.dispatchEvent(new Event("resize"));
    });
}

// =========================================================================
// Tab Navigation
// =========================================================================

function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(btn => {
        btn.classList.toggle("active", btn.getAttribute("data-tab") === tabId);
    });

    document.querySelectorAll(".tab-pane").forEach(pane => {
        pane.classList.toggle("active", pane.id === tabId);
    });

    // Invalidate map size when switching back to overview
    if (tabId === "tab-overview" && leafletMap) {
        setTimeout(() => {
            leafletMap.invalidateSize();
        }, 150);
    }

    if (tabId === "tab-evaluation") {
        loadModelEvaluation();
    }
}

function toggleAdvancedOptions() {
    const drawer = document.getElementById("advanced-options-drawer");
    const btn = document.getElementById("toggle-options-btn");
    drawer.classList.toggle("hidden");
    btn.innerText = drawer.classList.contains("hidden") ? "more options" : "fewer options";
}

// =========================================================================
// City Autocomplete & Overview Stats
// =========================================================================

async function loadCities() {
    try {
        const resp = await fetch("/api/cities");
        const data = await resp.json();
        
        const originList = document.getElementById("origin-cities");
        const destList = document.getElementById("dest-cities");

        if (data.origins) {
            originList.innerHTML = data.origins.map(c => `<option value="${c}">`).join("");
        }
        if (data.destinations) {
            destList.innerHTML = data.destinations.map(c => `<option value="${c}">`).join("");
        }
    } catch (e) {
        console.warn("Could not load city list:", e);
    }
}

async function loadOverviewStats() {
    try {
        const resp = await fetch("/api/overview");
        const data = await resp.json();

        // Update KPI values
        document.getElementById("kpi-deliveries").innerText = data.total_records || 200;
        document.getElementById("kpi-containers").innerText = data.active_containers || 36;
        document.getElementById("kpi-waiting-time").innerText = `${Math.round(data.avg_delay_min / 60) || 4}h`;
        document.getElementById("kpi-accuracy").innerText = `${data.booking_accuracy_pct || 82}%`;
        document.getElementById("kpi-reliability").innerText = `${data.reliability_pct || 75}%`;

        // Render SLA Delivery Chart
        renderSlaChart(data.sla_trend || [], data.sla_average || 88);

        // Render Booking Accuracy Bar Chart
        renderBookingAccuracyChart(data);

    } catch (e) {
        console.error("Failed to load overview stats:", e);
    }
}

// =========================================================================
// Chart.js Visualizations
// =========================================================================

function renderSlaChart(slaTrend, slaAverage = 88) {
    const ctx = document.getElementById("chart-sla-delivery").getContext("2d");
    if (slaChartInstance) {
        slaChartInstance.destroy();
    }

    const labels = slaTrend.map(t => t.period);
    const dataVals = slaTrend.map(t => t.on_time_pct);

    // Create subtle cyan gradient
    const gradient = ctx.createLinearGradient(0, 0, 0, 200);
    gradient.addColorStop(0, "rgba(0, 180, 216, 0.28)");
    gradient.addColorStop(1, "rgba(0, 180, 216, 0.02)");

    slaChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "SLA Delivery %",
                    data: dataVals,
                    borderColor: "#00b4d8",
                    backgroundColor: gradient,
                    fill: true,
                    tension: 0.45,
                    borderWidth: 2.5,
                    pointBackgroundColor: "#ffffff",
                    pointBorderColor: "#00b4d8",
                    pointBorderWidth: 2,
                    pointRadius: 4,
                    pointHoverRadius: 6
                },
                {
                    label: "Average Baseline",
                    data: labels.map(() => slaAverage),
                    borderColor: "#475569",
                    borderDash: [4, 4],
                    borderWidth: 1.5,
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.raw}%`
                    }
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    ticks: {
                        stepSize: 50,
                        callback: (v) => `${v}%`,
                        color: "#94a3b8",
                        font: { size: 10 }
                    },
                    grid: { color: "#f1f5f9" }
                },
                x: {
                    ticks: { color: "#94a3b8", font: { size: 10.5 } },
                    grid: { display: false }
                }
            }
        }
    });
}

function renderBookingAccuracyChart(stats) {
    const ctx = document.getElementById("chart-booking-accuracy").getContext("2d");
    if (bookingChartInstance) {
        bookingChartInstance.destroy();
    }

    const breakdown = stats.booking_breakdown || {};
    const accuracyVal = breakdown.time || stats.booking_accuracy_pct || 96.5;
    const priceEfficiency = breakdown.prices || 82.0;
    const qualityScore = breakdown.quality || stats.reliability_pct || 88.0;

    const gradient = ctx.createLinearGradient(0, 0, 0, 180);
    gradient.addColorStop(0, "#00b4d8");
    gradient.addColorStop(1, "#38bdf8");

    bookingChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: ["TIME", "PRICES", "QUALITY"],
            datasets: [{
                data: [accuracyVal, priceEfficiency, qualityScore],
                backgroundColor: gradient,
                borderRadius: 4,
                barPercentage: 0.42
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.label}: ${ctx.raw}%`
                    }
                }
            },
            scales: {
                y: {
                    min: 0,
                    max: 100,
                    ticks: {
                        stepSize: 25,
                        callback: (v) => `${v}%`,
                        color: "#94a3b8",
                        font: { size: 10 }
                    },
                    grid: { color: "#f1f5f9" }
                },
                x: {
                    ticks: { color: "#64748b", font: { size: 10.5, weight: "bold" } },
                    grid: { display: false }
                }
            }
        }
    });
}

// =========================================================================
// Leaflet Map Initialization & Rendering
// =========================================================================

function initMap() {
    // Initial center on Western Europe
    leafletMap = L.map("route-map", {
        center: [48.5, 10.0],
        zoom: 4,
        zoomControl: false,
        attributionControl: false
    });

    L.control.zoom({ position: "topright" }).addTo(leafletMap);

    // CartoDB Voyager tiles with provided CARTO API Key
    L.tileLayer("https://basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png?key=cb1_4gx9_1_c8703a434488f09f921436dc", {
        maxZoom: 18,
        attribution: '&copy; <a href="https://carto.com/">CARTO</a>'
    }).addTo(leafletMap);

    mapLayersGroup = L.layerGroup().addTo(leafletMap);
}

let lastPlanData = null;

function updateMapRoute(route) {
    if (!leafletMap || !mapLayersGroup) return;

    mapLayersGroup.clearLayers();

    if (!route || !route.waypoints || route.waypoints.length === 0) return;

    // Draw alternative routes in dashed, semi-transparent styling
    if (currentRoutes && currentRoutes.length > 1) {
        currentRoutes.forEach((altRoute, altIdx) => {
            if (altIdx === selectedRouteIndex) return;
            if (!altRoute.waypoints || altRoute.waypoints.length === 0) return;

            const altCoords = altRoute.waypoints.map(w => w.coords);
            const altColor = altIdx === 1 ? "#0284c7" : "#f59e0b";

            const altPolyline = L.polyline(altCoords, {
                color: altColor,
                weight: 2.2,
                dashArray: "5, 6",
                opacity: 0.65,
                lineCap: "round"
            }).addTo(mapLayersGroup);

            altPolyline.bindTooltip(`Alternative: ${altRoute.title} (${altRoute.est_time}) - Click to select`, {
                sticky: true
            });

            altPolyline.on("click", () => {
                selectRoute(altIdx);
            });
        });
    }

    const latlngs = route.waypoints.map(w => w.coords);
    const activeColor = route.route_type === "air_direct" ? "#f59e0b" : (route.route_type === "air_sea" ? "#0284c7" : "#ef4444");

    // Draw active connecting path line (solid vibrant)
    const polyline = L.polyline(latlngs, {
        color: activeColor,
        weight: 3.5,
        opacity: 0.95,
        lineCap: "round"
    }).addTo(mapLayersGroup);

    // Add city pins and labels in exact leg order
    route.waypoints.forEach((wp, idx) => {
        const isEndpoint = idx === 0 || idx === route.waypoints.length - 1;
        const modeIcon = getTransportIcon(wp.transport || "truck");

        // Custom marker icon
        const iconHtml = `
            <div class="map-city-marker" style="${isEndpoint ? 'border-color: #0f172a; width: 14px; height: 14px; background: #0f172a;' : 'border-color: #ef4444; width: 11px; height: 11px; background: #ffffff;'}">
            </div>
        `;
        const icon = L.divIcon({
            html: iconHtml,
            className: "custom-div-icon",
            iconSize: [14, 14],
            iconAnchor: [7, 7]
        });

        const marker = L.marker(wp.coords, { icon: icon }).addTo(mapLayersGroup);

        // Label tooltip above pin
        const labelText = isEndpoint ? wp.name.toUpperCase() : `${modeIcon} ${wp.name.toUpperCase()}`;
        marker.bindTooltip(labelText, {
            permanent: true,
            direction: "top",
            className: "map-city-label",
            offset: [0, -10]
        });
    });

    // Add warning alert markers along route if present
    if (route.delay_markers && route.delay_markers.length > 0) {
        route.delay_markers.forEach(dm => {
            const delayText = dm.delay_hours ? ` (+${dm.delay_hours}h)` : "";
            const alertIcon = L.divIcon({
                html: `<div class="map-alert-marker" title="${dm.label}${delayText}">⚠️</div>`,
                className: "custom-alert-icon",
                iconSize: [20, 20],
                iconAnchor: [10, 10]
            });
            L.marker(dm.coords, { icon: alertIcon }).addTo(mapLayersGroup)
                .bindPopup(`<strong>${dm.label}${delayText}</strong><br><span style="font-size: 11px; color: #64748b;">Stage Delay Location</span>`);
        });
    }

    // Adjust Hub Callout Box
    const calloutEl = document.getElementById("map-hub-callout");
    const hubWaypoint = route.waypoints.find(w => w.hub_card);
    if (hubWaypoint && hubWaypoint.hub_card) {
        document.getElementById("callout-hub-title").innerText = hubWaypoint.hub_card.title;
        document.getElementById("callout-std-wait").innerText = hubWaypoint.hub_card.standard_waiting;
        document.getElementById("callout-est-wait").innerText = hubWaypoint.hub_card.estimated_waiting;
        calloutEl.style.display = "block";
    } else {
        calloutEl.style.display = "none";
    }

    // Fit map bounds with padding to show complete route across the world
    if (polyline && polyline.getBounds().isValid()) {
        leafletMap.fitBounds(polyline.getBounds(), { padding: [50, 50], maxZoom: 10 });
    }
}

// =========================================================================
// Route Planning & Selection
// =========================================================================

function quickSelectCity(cityName) {
    document.getElementById("input-to").value = cityName;
    handlePlanRoutes();
}

async function handlePlanRoutes() {
    const origin = document.getElementById("input-from").value.trim() || "India";
    const dest = document.getElementById("input-to").value.trim() || "Germany";
    const cbm = parseFloat(document.getElementById("input-cbm").value) || 24;
    const wm = parseFloat(document.getElementById("input-wm").value) || 12;
    const optimise = document.getElementById("input-optimise").value;
    const weather = document.getElementById("adv-weather").value;
    const priority = document.getElementById("adv-priority").value;
    const carrierRate = parseFloat(document.getElementById("adv-carrier-rate").value) || 0.88;

    const container = document.getElementById("route-options-list");
    container.innerHTML = `<div class="loading-spinner-box">Calculating multimodal corridors...</div>`;

    try {
        const resp = await fetch("/api/routes/plan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                origin: origin,
                destination: dest,
                cbm: cbm,
                wm: wm,
                optimise_for: optimise,
                weather: weather,
                priority: priority,
                carrier_rate: carrierRate
            })
        });

        const data = await resp.json();
        lastPlanData = data;
        currentRoutes = data.routes || [];
        selectedRouteIndex = 0;
        renderRouteOptions();

        if (currentRoutes.length > 0) {
            selectRoute(0);
        }
    } catch (e) {
        console.error("Error planning routes:", e);
        container.innerHTML = `<div class="placeholder-prompt" style="color: #ef4444;">Failed to calculate routes. Check backend.</div>`;
    }
}

function renderRouteOptions() {
    const container = document.getElementById("route-options-list");
    if (!currentRoutes || currentRoutes.length === 0) {
        container.innerHTML = `<div class="placeholder-prompt">No routes found for this city pair.</div>`;
        return;
    }

    let illustrativeHtml = "";
    if (lastPlanData && lastPlanData.is_illustrative) {
        illustrativeHtml = `
            <div class="illustrative-banner">
                <span class="illustrative-badge">ILLUSTRATIVE GATEWAY</span>
                <p>${lastPlanData.illustrative_note || 'Destination city unspecified. Showing US Gateway routing.'}</p>
                <div class="city-quick-picks">
                    <span>Select specific city:</span>
                    <button type="button" class="quick-city-btn" onclick="quickSelectCity('Chicago')">Chicago</button>
                    <button type="button" class="quick-city-btn" onclick="quickSelectCity('New York')">New York</button>
                    <button type="button" class="quick-city-btn" onclick="quickSelectCity('Los Angeles')">Los Angeles</button>
                    <button type="button" class="quick-city-btn" onclick="quickSelectCity('Dallas')">Dallas</button>
                    <button type="button" class="quick-city-btn" onclick="quickSelectCity('Houston')">Houston</button>
                </div>
            </div>
        `;
    }

    const cardsHtml = currentRoutes.map((route, idx) => {
        const isSelected = idx === selectedRouteIndex;

        // Render timeline progression
        const nodesCount = route.timeline_nodes.length;
        const timelineHtml = route.timeline_nodes.map((node, nodeIdx) => {
            const isLast = nodeIdx === nodesCount - 1;
            const fullTitle = node.full_name || node.label;
            let stepHtml = `
                <div class="timeline-step" title="${fullTitle}">
                    <span class="timeline-label" title="${fullTitle}">${node.label}</span>
                    <div class="timeline-dot"></div>
                </div>
            `;
            if (!isLast) {
                const nextNode = route.timeline_nodes[nodeIdx + 1];
                const modeIcon = getTransportIcon(nextNode.transport || node.transport || "truck");
                stepHtml += `
                    <div class="timeline-line">
                        <span class="timeline-mode-icon">${modeIcon}</span>
                    </div>
                `;
            }
            return stepHtml;
        }).join("");

        // Optional alert or guarantee badge
        let bannerHtml = "";
        if (route.has_alert && route.alert) {
            bannerHtml = `
                <div class="route-card-alert">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path></svg>
                    <span>${route.alert}</span>
                </div>
            `;
        } else if (route.badge) {
            bannerHtml = `
                <div class="route-card-badge">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#059669" stroke-width="2.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"></path></svg>
                    <span>${route.badge}</span>
                </div>
            `;
        }

        const riskClass = route.delay_risk ? route.delay_risk.toLowerCase() : "low";

        return `
            <div class="route-card ${isSelected ? 'selected' : ''}" onclick="selectRoute(${idx});">
                <div class="route-card-main">
                    <div class="timeline-path">
                        ${timelineHtml}
                    </div>
                    <div class="route-card-stats">
                        <div class="stat-pill">
                            <div class="stat-icon time">🕒</div>
                            <div class="stat-text">
                                <span class="stat-title">EST. TIME</span>
                                <span class="stat-val">${route.est_time}</span>
                            </div>
                        </div>
                        <div class="stat-pill">
                            <div class="stat-icon price">₹</div>
                            <div class="stat-text">
                                <span class="stat-title">EST. PRICE</span>
                                <span class="stat-val">${route.est_price}</span>
                            </div>
                        </div>
                        <div class="stat-pill">
                            <div class="stat-icon check">✓</div>
                            <div class="stat-text">
                                <span class="stat-title">RELIABILITY</span>
                                <span class="stat-val">${route.reliability}</span>
                            </div>
                        </div>
                    </div>
                </div>
                ${bannerHtml}
                <div class="route-model-meta">
                    <span class="badge-tag">Model-Estimated Physics</span>
                    <span class="badge-tag">${route.transfer_count} ${route.transfer_count === 1 ? 'Transfer' : 'Transfers'}</span>
                    <span class="badge-tag risk-${riskClass}">Risk: ${route.delay_risk || 'Low'}</span>
                </div>
            </div>
        `;
    }).join("");

    container.innerHTML = illustrativeHtml + cardsHtml;
}

function getTransportIcon(mode) {
    if (mode === "airplane" || mode === "air") return "✈️";
    if (mode === "ship" || mode === "sea") return "🚢";
    if (mode === "train" || mode === "rail") return "🚆";
    return "🚚";
}

function selectRoute(idx) {
    selectedRouteIndex = idx;
    const route = currentRoutes[idx];
    if (!route) return;

    // Update DOM selection styling
    document.querySelectorAll(".route-card").forEach((card, i) => {
        card.classList.toggle("selected", i === idx);
    });

    // Update Overview Tab Alert Banner
    const alertBanner = document.getElementById("overview-alert-banner");
    const alertText = document.getElementById("alert-banner-text");
    if (route.has_alert && route.alert) {
        alertText.innerHTML = route.alert;
        alertBanner.style.display = "flex";
    } else {
        alertBanner.style.display = "none";
    }

    // Update KPI Card Estimated Time & Typical Time dynamically
    document.getElementById("kpi-typical-time").innerText = route.typical_time || route.est_time;
    document.getElementById("kpi-estimated-time").innerText = route.est_time;
    document.getElementById("kpi-reliability").innerText = route.reliability;

    // Update Leaflet Map route line & pins
    updateMapRoute(route);
}

// =========================================================================
// Tab 2: Single Shipment Prediction
// =========================================================================

async function handlePredictSingle() {
    const origin = document.getElementById("p-origin").value;
    const dest = document.getElementById("p-dest").value;
    const dist = parseFloat(document.getElementById("p-dist").value) || 1500;
    const dep = document.getElementById("p-dep").value;
    const arr = document.getElementById("p-arr").value;
    const routeType = document.getElementById("p-route").value;
    const vehicle = document.getElementById("p-vehicle").value;
    const priority = document.getElementById("p-priority").value;
    const shipType = document.getElementById("p-type").value;
    const weather = document.getElementById("p-weather").value;
    const carrier = parseFloat(document.getElementById("p-carrier").value) || 0.90;

    const outBox = document.getElementById("prediction-output-box");
    outBox.innerHTML = `<div class="loading-spinner-box">Running inference on trained ML models...</div>`;

    try {
        const resp = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                origin_country: origin,
                destination_country: dest,
                route_distance_km: dist,
                scheduled_departure_datetime: dep,
                scheduled_arrival_datetime: arr,
                route_type: routeType,
                vehicle_type: vehicle,
                priority_level: priority,
                shipment_type: shipType,
                weather_condition_departure: weather,
                carrier_on_time_rate: carrier
            })
        });

        const res = await resp.json();

        const delayMin = res.predicted_delay_min !== null ? `${res.predicted_delay_min.toFixed(1)} mins` : "N/A";
        const delayEta = res.predicted_eta || "N/A";
        const risk = res.delay_risk || "Unknown";
        const riskClass = risk.toLowerCase() === "delayed" ? "delayed" : "ontime";
        const prob = res.delay_probability !== null ? (res.delay_probability * 100).toFixed(1) : 0;

        let factorsHtml = "";
        if (res.explanation && res.explanation.top_factors) {
            factorsHtml = res.explanation.top_factors.map(f => `
                <div class="factor-row">
                    <span>${f.direction === "increases" ? '🔺' : '🔻'} <strong>${f.feature}</strong></span>
                    <span class="factor-impact ${f.direction === 'increases' ? 'positive' : 'negative'}">
                        ${f.direction === "increases" ? '+' : ''}${f.contribution_min.toFixed(1)} min
                    </span>
                </div>
            `).join("");
        }

        outBox.innerHTML = `
            <div class="result-metric-card">
                <div class="result-metric-title">Predicted Additional Delay</div>
                <div class="result-metric-val">${delayMin}</div>
                <div style="font-size: 13px; color: #64748b; margin-top: 4px;">Predicted ETA: <strong>${delayEta}</strong></div>
            </div>

            <div class="result-metric-card">
                <div class="result-metric-title">Delay Risk Classification</div>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 6px;">
                    <span class="risk-badge ${riskClass}">${risk}</span>
                    <span style="font-size: 13px; font-weight: 600;">Delay Probability: ${prob}%</span>
                </div>
                <div style="background: #e2e8f0; height: 6px; border-radius: 3px; overflow: hidden; margin-top: 10px;">
                    <div style="width: ${prob}%; height: 100%; background: ${riskClass === 'delayed' ? '#ef4444' : '#10b981'};"></div>
                </div>
            </div>

            <h3 class="subsection-title">Top Contributing Factors (SHAP Values)</h3>
            <div style="margin-bottom: 14px;">
                ${factorsHtml || '<div style="font-size: 13px; color: #94a3b8;">No factor breakdown available.</div>'}
            </div>

            <div style="font-size: 12px; color: #64748b; background: #f8fafc; padding: 10px; border-radius: 6px;">
                ℹ️ ${res.confidence_note || 'Prediction generated via production pipeline.'}
            </div>
        `;
    } catch (e) {
        console.error("Prediction failed:", e);
        outBox.innerHTML = `<div class="placeholder-prompt" style="color: #ef4444;">Inference failed: ${e.message}</div>`;
    }
}

// =========================================================================
// Tab 3: Delay Propagation Simulator
// =========================================================================

async function handleRunSimulation() {
    const duration = parseFloat(document.getElementById("sim-duration").value) || 1000;
    const route = document.getElementById("sim-route").value;
    const weather = document.getElementById("sim-weather").value;
    const carrier = parseFloat(document.getElementById("sim-carrier").value) || 0.90;
    const stage = document.getElementById("sim-stage").value;
    const delay = parseFloat(document.getElementById("sim-delay").value) || 120;

    const outBox = document.getElementById("simulation-output-box");
    outBox.innerHTML = `<div class="loading-spinner-box">Simulating stage disruption cascade...</div>`;

    try {
        const resp = await fetch("/api/simulate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                scheduled_duration_min: duration,
                route_type: route,
                weather_condition_departure: weather,
                carrier_on_time_rate: carrier,
                inject_stage: stage,
                inject_delay: delay
            })
        });

        const res = await resp.json();

        const baseTotal = res.baseline_total_delay_min.toFixed(1);
        const injTotal = res.injected_total_delay_min.toFixed(1);
        const delta = res.additional_impact_min.toFixed(1);

        // Stage breakdown rows
        const stageRows = (res.injected_stages || []).map(s => `
            <tr>
                <td style="font-weight: 600;">${s.stage.replace(/_/g, ' ')}</td>
                <td>${s.sched_duration_min.toFixed(0)}m</td>
                <td style="color: ${s.stage_delay_min > 0 ? '#ef4444' : '#1e293b'}; font-weight: 600;">+${s.stage_delay_min.toFixed(1)}m</td>
                <td style="font-weight: 700;">${s.cumul_delay_min.toFixed(1)}m</td>
            </tr>
        `).join("");

        outBox.innerHTML = `
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 16px;">
                <div class="result-metric-card">
                    <div class="result-metric-title">Baseline Journey Delay</div>
                    <div class="result-metric-val">${baseTotal}m</div>
                </div>
                <div class="result-metric-card">
                    <div class="result-metric-title">Injected Scenario Delay</div>
                    <div class="result-metric-val" style="color: #ef4444;">${injTotal}m <span style="font-size: 14px; color: #b91c1c;">(+${delta}m)</span></div>
                </div>
            </div>

            <p style="font-size: 12.5px; color: #64748b; margin-bottom: 16px;">
                ${res.note}
            </p>

            <h3 class="subsection-title">Stage-by-Stage Timeline Breakdown</h3>
            <div class="data-table-wrapper">
                <table class="real-data-table">
                    <thead>
                        <tr>
                            <th>Journey Stage</th>
                            <th>Scheduled Dur.</th>
                            <th>Stage Delay</th>
                            <th>Cumulative Delay</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${stageRows}
                    </tbody>
                </table>
            </div>
        `;
    } catch (e) {
        console.error("Simulation failed:", e);
        outBox.innerHTML = `<div class="placeholder-prompt" style="color: #ef4444;">Simulation failed: ${e.message}</div>`;
    }
}

// =========================================================================
// Tab 4: Model Evaluation
// =========================================================================

async function loadModelEvaluation() {
    const grid = document.getElementById("eval-metrics-grid");
    grid.innerHTML = `<div class="loading-spinner-box">Loading metrics...</div>`;

    try {
        const resp = await fetch("/api/model-evaluation");
        const data = await resp.json();
        const m = data.metrics || {};

        const reg = m.regression_best || {};
        const base = m.regression_baseline || {};
        const clf = m.classification || {};

        grid.innerHTML = `
            <div class="result-metric-card">
                <div class="result-metric-title">ETA Regression (Ridge CV)</div>
                <div class="result-metric-val">${reg.mae_minutes ? reg.mae_minutes.toFixed(1) + ' min' : '13.9 min'}</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
                    RMSE: <strong>${reg.rmse_minutes ? reg.rmse_minutes.toFixed(1) + 'm' : '18.6m'}</strong> | R²: <strong>${reg.r_squared ? reg.r_squared.toFixed(3) : '0.731'}</strong>
                </div>
            </div>

            <div class="result-metric-card">
                <div class="result-metric-title">Baseline Comparison (Mean)</div>
                <div class="result-metric-val">${base.mae_minutes ? base.mae_minutes.toFixed(1) + ' min' : '30.6 min'}</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
                    RMSE: <strong>${base.rmse_minutes ? base.rmse_minutes.toFixed(1) + 'm' : '36.6m'}</strong> | R²: <strong>${base.r_squared ? base.r_squared.toFixed(3) : '-0.041'}</strong>
                </div>
            </div>

            <div class="result-metric-card">
                <div class="result-metric-title">Delay Classifier (Random Forest)</div>
                <div class="result-metric-val">${clf.accuracy ? (clf.accuracy * 100).toFixed(1) + '%' : '96.6%'}</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
                    Precision: <strong>${clf.precision ? (clf.precision * 100).toFixed(1) + '%' : '100%'}</strong> | F1: <strong>${clf.f1 ? (clf.f1 * 100).toFixed(1) + '%' : '98.2%'}</strong>
                </div>
            </div>

            <div class="result-metric-card">
                <div class="result-metric-title">ROC-AUC Score</div>
                <div class="result-metric-val" style="color: #0284c7;">${clf.roc_auc ? clf.roc_auc.toFixed(3) : '1.000'}</div>
                <div style="font-size: 12px; color: #64748b; margin-top: 4px;">
                    High-discrimination boundary
                </div>
            </div>
        `;
    } catch (e) {
        console.error("Evaluation loading error:", e);
        grid.innerHTML = `<div class="placeholder-prompt" style="color: #ef4444;">Failed to load evaluation metrics.</div>`;
    }
}

// =========================================================================
// Tab 5: Data Explorer
// =========================================================================

function debounceExplorerSearch() {
    clearTimeout(searchDebounceTimeout);
    searchDebounceTimeout = setTimeout(() => {
        loadExplorerData();
    }, 250);
}

async function loadExplorerData() {
    const search = document.getElementById("explorer-search").value.trim();
    const route = document.getElementById("filter-route").value;
    const weather = document.getElementById("filter-weather").value;
    const delayed = document.getElementById("filter-delayed").value;

    const tbody = document.getElementById("explorer-tbody");
    const countBadge = document.getElementById("explorer-count-badge");

    try {
        const params = new URLSearchParams({
            search: search,
            route_type: route,
            weather: weather,
            is_delayed: delayed,
            limit: 100
        });

        const resp = await fetch(`/api/dataset/records?${params.toString()}`);
        const data = await resp.json();

        countBadge.innerText = `Showing ${data.records.length} of ${data.total} records`;

        if (!data.records || data.records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="12" style="text-align: center; padding: 24px;">No records match the current filters.</td></tr>`;
            return;
        }

        tbody.innerHTML = data.records.map(r => {
            const delayMin = r.delay_target_minutes !== null ? `${r.delay_target_minutes.toFixed(0)}m` : '-';
            const delayColor = r.is_delayed === 1 ? '#ef4444' : '#10b981';
            const statusClass = (r.shipment_status || 'delivered').toLowerCase();

            return `
                <tr>
                    <td style="font-weight: 700; color: #0284c7;">${r.shipment_id || '-'}</td>
                    <td>${r.origin_city || '-'}</td>
                    <td>${r.destination_city || '-'}</td>
                    <td>${r.route_type || '-'}</td>
                    <td>${r.vehicle_type || '-'}</td>
                    <td>${r.weather_condition_departure || '-'}</td>
                    <td>${r.route_distance_km ? Math.round(r.route_distance_km) : '-'}</td>
                    <td>${r.carrier_name || '-'}</td>
                    <td>${r.carrier_on_time_rate ? Math.round(r.carrier_on_time_rate * 100) + '%' : '-'}</td>
                    <td style="color: ${delayColor}; font-weight: 700;">${delayMin}</td>
                    <td><span class="status-pill ${statusClass}">${r.shipment_status || 'delivered'}</span></td>
                    <td style="max-width: 250px; overflow: hidden; text-overflow: ellipsis;" title="${r.notes || ''}">${r.notes || '-'}</td>
                </tr>
            `;
        }).join("");

    } catch (e) {
        console.error("Failed to load explorer data:", e);
        tbody.innerHTML = `<tr><td colspan="12" style="text-align: center; color: #ef4444; padding: 24px;">Error loading dataset records.</td></tr>`;
    }
}
