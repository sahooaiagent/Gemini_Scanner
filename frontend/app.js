/* ═══════════════════════════════════════════════════════════════════
   GEMINI SCANNER — ENTERPRISE APPLICATION LOGIC
   ═══════════════════════════════════════════════════════════════════ */

const API_URL = 'http://localhost:8000';
let allResults = [];
let scanRunning = false;
let logPollInterval = null;
let currentSort = { col: null, asc: true };

// ── DOM REFS ──
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ══════════════════════════════════════════════════════════════
// INITIALIZATION
// ══════════════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
    initNavigation();
    initClock();
    initTickerTape();
    initScannerControls();
    initFilterControls();
    initChartModal();
    initMobileMenu();
    setConnectionStatus(true);

    // Initial data fetch
    fetchMarketData();
    fetchResults();

    // Periodic refresh
    setInterval(fetchMarketData, 60000);
    setInterval(() => { if (!scanRunning) fetchResults(); }, 30000);
});

// ══════════════════════════════════════════════════════════════
// NAVIGATION
// ══════════════════════════════════════════════════════════════
function initNavigation() {
    $$('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            const tab = link.dataset.tab;
            $$('.nav-link').forEach(l => l.classList.remove('active'));
            link.classList.add('active');
            $$('.tab-content').forEach(t => t.classList.remove('active'));
            $(`#tab-${tab}`).classList.add('active');
            // Close mobile sidebar
            $('#sidebar').classList.remove('open');
        });
    });
}

// ══════════════════════════════════════════════════════════════
// LIVE CLOCK
// ══════════════════════════════════════════════════════════════
function initClock() {
    const clockEl = $('#liveClock');
    function updateClock() {
        const now = new Date();
        clockEl.textContent = now.toLocaleString('en-US', {
            hour: '2-digit', minute: '2-digit', second: '2-digit',
            hour12: false, day: '2-digit', month: 'short', year: 'numeric'
        });
    }
    updateClock();
    setInterval(updateClock, 1000);
}

// ══════════════════════════════════════════════════════════════
// TICKER TAPE
// ══════════════════════════════════════════════════════════════
const INDICES = [
    { name: 'NIFTY 50',   ticker: '^NSEI',     price: '—', change: '—' },
    { name: 'BANK NIFTY', ticker: '^NSEBANK',   price: '—', change: '—' },
    { name: 'DOW JONES',  ticker: '^DJI',       price: '—', change: '—' },
    { name: 'NASDAQ',     ticker: '^IXIC',      price: '—', change: '—' },
];

function initTickerTape() {
    renderTickerTape();
}

function renderTickerTape() {
    const track = $('#tickerTrack');
    // Duplicate items for seamless loop
    const items = [...INDICES, ...INDICES, ...INDICES].map(idx => {
        const changeCls = parseFloat(idx.change) >= 0 ? 'change-up' : 'change-down';
        const arrow = parseFloat(idx.change) >= 0 ? '▲' : '▼';
        return `
            <span class="ticker-item">
                <span class="name">${idx.name}</span>
                <span class="price">${idx.price}</span>
                <span class="${changeCls}">${arrow} ${idx.change}</span>
            </span>
        `;
    }).join('');
    track.innerHTML = items;
}

async function fetchMarketData() {
    // Try to get market data from backend — fallback to static if unavailable
    try {
        const res = await fetch(`${API_URL}/api/market-data`);
        if (res.ok) {
            const data = await res.json();
            if (data.indices) {
                data.indices.forEach(d => {
                    const idx = INDICES.find(i => i.name === d.name);
                    if (idx) {
                        idx.price = d.price;
                        idx.change = d.change;
                    }
                });
                renderTickerTape();
                renderHeatmap();
            }
        }
    } catch (e) {
        // Backend market-data endpoint may not exist yet — that's fine
        renderHeatmap();
    }
}

// ══════════════════════════════════════════════════════════════
// HEATMAP
// ══════════════════════════════════════════════════════════════
function renderHeatmap() {
    const grid = $('#heatmapGrid');
    grid.innerHTML = INDICES.map(idx => {
        const changeVal = parseFloat(idx.change) || 0;
        const cls = changeVal > 0 ? 'positive' : changeVal < 0 ? 'negative' : 'neutral';
        const changeCls = changeVal >= 0 ? 'up' : 'down';
        const arrow = changeVal >= 0 ? '▲' : '▼';
        return `
            <div class="heatmap-tile ${cls}">
                <span class="heatmap-name">${idx.name}</span>
                <span class="heatmap-price">${idx.price}</span>
                <span class="heatmap-change ${changeCls}">
                    <i class="fas fa-caret-${changeVal >= 0 ? 'up' : 'down'}"></i>
                    ${idx.change}%
                </span>
            </div>
        `;
    }).join('');
}

// ══════════════════════════════════════════════════════════════
// FETCH & RENDER RESULTS
// ══════════════════════════════════════════════════════════════
async function fetchResults() {
    try {
        const res = await fetch(`${API_URL}/api/results`);
        if (!res.ok) return;
        const data = await res.json();
        allResults = data.results || [];
        if (data.scan_time) {
            updateLastScanTime(data.scan_time);
        }
        renderResults();
        updateStats();
    } catch (e) {
        // API may not have /api/results yet
    }
}

function renderResults() {
    const body = $('#signalsBody');
    const empty = $('#emptyState');
    const countEl = $('#resultCount');
    const searchVal = ($('#searchInput').value || '').toLowerCase();
    const signalFilter = $('.chip.active')?.dataset?.filter || 'all';
    const tfFilter = $('#tfFilter').value;

    let filtered = allResults.filter(r => {
        if (searchVal && !r['Crypto Name']?.toLowerCase().includes(searchVal)) return false;
        if (signalFilter !== 'all' && r.Signal !== signalFilter) return false;
        if (tfFilter !== 'all' && r.Timeperiod !== tfFilter) return false;
        return true;
    });

    // Sort
    if (currentSort.col) {
        filtered.sort((a, b) => {
            let va = a[currentSort.col] || '';
            let vb = b[currentSort.col] || '';
            if (typeof va === 'string') va = va.toLowerCase();
            if (typeof vb === 'string') vb = vb.toLowerCase();
            if (va < vb) return currentSort.asc ? -1 : 1;
            if (va > vb) return currentSort.asc ? 1 : -1;
            return 0;
        });
    }

    if (filtered.length === 0) {
        body.innerHTML = '';
        empty.style.display = 'block';
        countEl.textContent = '0 results';
        return;
    }

    empty.style.display = 'none';
    countEl.textContent = `${filtered.length} result${filtered.length !== 1 ? 's' : ''}`;

    body.innerHTML = filtered.map((r, i) => {
        const sigCls = r.Signal === 'LONG' ? 'long' : 'short';
        const sigIcon = r.Signal === 'LONG' ? 'fa-arrow-up' : 'fa-arrow-down';
        const changeStr = r['Daily Change'] || '—';
        const changeVal = parseFloat(changeStr);
        const changeCls = isNaN(changeVal) ? '' : (changeVal >= 0 ? 'change-positive' : 'change-negative');
        const name = r['Crypto Name'] || '—';
        const tfMap = { '15min': '15m', '30min': '30m', '45min': '45m', '1hr': '1h', '2hr': '2h', '4hr': '4h', '1 day': '1D', '1 week': '1W' };
        const tfDisplay = tfMap[r.Timeperiod] || r.Timeperiod;

        return `
            <tr style="animation: fadeUp 0.3s ${0.03 * i}s var(--ease-out) both">
                <td><strong>${name}</strong></td>
                <td><span class="tf-badge">${tfDisplay}</span></td>
                <td>
                    <span class="signal-badge ${sigCls}">
                        <i class="fas ${sigIcon}"></i>
                        ${r.Signal}
                    </span>
                </td>
                <td class="mono">${r.Angle || '—'}</td>
                <td class="${changeCls}">${changeStr}</td>
                <td class="mono">${r.Timestamp || '—'}</td>
                <td>
                    <button class="chart-btn" onclick="openChart('${name}', '${r.Timeperiod}')">
                        <i class="fas fa-chart-candlestick"></i> Chart
                    </button>
                </td>
            </tr>
        `;
    }).join('');
}

function updateStats() {
    const total = allResults.length;
    const longs = allResults.filter(r => r.Signal === 'LONG').length;
    const shorts = allResults.filter(r => r.Signal === 'SHORT').length;
    animateCounter('totalSignals', total);
    animateCounter('longSignals', longs);
    animateCounter('shortSignals', shorts);
}

function updateLastScanTime(timeStr) {
    if (!timeStr) return;
    const scanDate = new Date(timeStr);
    const now = new Date();
    const diffMs = now - scanDate;
    const diffMin = Math.floor(diffMs / 60000);

    let display;
    if (diffMin < 1) display = 'Just now';
    else if (diffMin < 60) display = `${diffMin}m ago`;
    else if (diffMin < 1440) display = `${Math.floor(diffMin / 60)}h ago`;
    else display = `${Math.floor(diffMin / 1440)}d ago`;

    $('#lastScanTime').textContent = display;
    $('#lastScanTime').title = `Last scan: ${timeStr}`;
}

function animateCounter(id, target) {
    const el = document.getElementById(id);
    if (!el) return;
    const current = parseInt(el.textContent) || 0;
    if (current === target) return;
    const duration = 600;
    const start = performance.now();
    function step(ts) {
        const progress = Math.min((ts - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(current + (target - current) * eased);
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

// ══════════════════════════════════════════════════════════════
// SCANNER CONTROLS
// ══════════════════════════════════════════════════════════════
function initScannerControls() {
    // Index chip toggles
    $$('.index-chip').forEach(chip => {
        chip.addEventListener('click', () => chip.classList.toggle('active'));
    });

    // Timeframe chip toggles
    $$('.tf-chip').forEach(chip => {
        chip.addEventListener('click', () => chip.classList.toggle('active'));
    });

    // Run scan button
    $('#runScanBtn').addEventListener('click', runScan);

    // Refresh button
    $('#refreshBtn').addEventListener('click', () => {
        fetchResults();
        fetchMarketData();
        showToast('Data refreshed', 'info');
    });

    // Export CSV
    $('#exportCsvBtn').addEventListener('click', exportCSV);

    // Clear logs
    $('#clearLogsBtn').addEventListener('click', () => {
        $('#logOutput').innerHTML = `
            <div class="log-line system">
                <span class="log-ts">system</span>
                <span class="log-msg">Logs cleared</span>
            </div>
        `;
    });
}

async function runScan() {
    if (scanRunning) return;

    const indices = Array.from($$('.index-chip.active')).map(c => c.dataset.index);
    const timeframes = Array.from($$('.tf-chip.active')).map(c => c.dataset.tf);

    if (indices.length === 0) {
        showToast('Select at least one index', 'warning');
        return;
    }
    if (timeframes.length === 0) {
        showToast('Select at least one timeframe', 'warning');
        return;
    }

    scanRunning = true;
    const btn = $('#runScanBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Scanning...</span>';
    btn.classList.remove('pulse-glow');

    // Show progress bar
    const progressWrap = $('#scanProgressWrap');
    progressWrap.classList.remove('hidden');
    updateProgress(0, 'Starting scan...');

    // Clear previous results
    allResults = [];
    renderResults();
    updateStats();

    // Add scan start log
    addLogLine('info', `🔄 SCAN IN PROGRESS — Indices: ${indices.join(', ')} | TFs: ${timeframes.join(', ')}`);

    // Start log polling
    startLogPolling();

    // Simulate progress
    let prog = 0;
    const totalSteps = indices.length * timeframes.length;
    const progInterval = setInterval(() => {
        if (!scanRunning) { clearInterval(progInterval); return; }
        prog = Math.min(prog + (90 / totalSteps / 2), 90);
        updateProgress(prog, `Analyzing ${indices.length} indices across ${timeframes.length} timeframes...`);
    }, 500);

    try {
        const res = await fetch(`${API_URL}/api/scan`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ indices, timeframes })
        });

        clearInterval(progInterval);

        if (!res.ok) throw new Error(`Server error: ${res.status}`);

        const data = await res.json();
        allResults = data.data || [];
        
        updateProgress(100, 'Scan complete!');
        addLogLine('success', `✅ SCAN COMPLETED — ${allResults.length} signal(s) found`);

        renderResults();
        updateStats();
        updateLastScanTime(new Date().toISOString());

        showToast(`Scan complete! ${allResults.length} signal(s) found.`, 'success');

        // Switch to dashboard tab to show results
        if (allResults.length > 0) {
            setTimeout(() => {
                $$('.nav-link').forEach(l => l.classList.remove('active'));
                $('#nav-dashboard').classList.add('active');
                $$('.tab-content').forEach(t => t.classList.remove('active'));
                $('#tab-dashboard').classList.add('active');
            }, 800);
        }

    } catch (err) {
        clearInterval(progInterval);
        updateProgress(0, 'Scan failed');
        addLogLine('error', `❌ Scan failed: ${err.message}`);
        showToast(`Scan failed: ${err.message}`, 'error');
    } finally {
        scanRunning = false;
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play"></i> <span>Run Scanner</span>';
        btn.classList.add('pulse-glow');
        stopLogPolling();

        setTimeout(() => {
            progressWrap.classList.add('hidden');
        }, 3000);
    }
}

function updateProgress(pct, detail) {
    $('#scanProgressBar').style.width = `${pct}%`;
    $('#scanProgressPct').textContent = `${Math.round(pct)}%`;
    if (detail) $('#scanProgressDetail').textContent = detail;
}

// ══════════════════════════════════════════════════════════════
// LOG POLLING
// ══════════════════════════════════════════════════════════════
function startLogPolling() {
    stopLogPolling();
    logPollInterval = setInterval(fetchLogs, 2000);
}
function stopLogPolling() {
    if (logPollInterval) { clearInterval(logPollInterval); logPollInterval = null; }
}

async function fetchLogs() {
    try {
        const res = await fetch(`${API_URL}/api/logs`);
        if (!res.ok) return;
        const data = await res.json();
        const logEl = $('#logOutput');

        if (data.logs && data.logs.length > 0) {
            logEl.innerHTML = data.logs.map(line => {
                const trimmed = (typeof line === 'string' ? line : '').trim();
                if (!trimmed) return '';
                let cls = '';
                if (trimmed.includes('ERROR') || trimmed.includes('error')) cls = 'error';
                else if (trimmed.includes('WARNING') || trimmed.includes('warning')) cls = 'warn';
                else if (trimmed.includes('SIGNAL') || trimmed.includes('✅') || trimmed.includes('COMPLETED')) cls = 'success';
                else if (trimmed.includes('INFO') || trimmed.includes('🔄') || trimmed.includes('Scanning') || trimmed.includes('Fetching')) cls = 'info';

                // Extract timestamp
                const tsMatch = trimmed.match(/^(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})/);
                const ts = tsMatch ? tsMatch[1].split(' ')[1] : '';
                const msg = tsMatch ? trimmed.slice(tsMatch[0].length).replace(/^\s*-\s*/, '').replace(/^\s*INFO\s*-\s*/, '').replace(/^\s*ERROR\s*-\s*/, '').trim() : trimmed;

                return `<div class="log-line ${cls}"><span class="log-ts">${ts}</span><span class="log-msg">${msg}</span></div>`;
            }).join('');
            logEl.scrollTop = logEl.scrollHeight;
        }
    } catch (e) { /* Silently fail */ }
}

function addLogLine(cls, msg) {
    const logEl = $('#logOutput');
    const now = new Date();
    const ts = now.toTimeString().slice(0, 8);
    logEl.innerHTML += `<div class="log-line ${cls}"><span class="log-ts">${ts}</span><span class="log-msg">${msg}</span></div>`;
    logEl.scrollTop = logEl.scrollHeight;
}

// ══════════════════════════════════════════════════════════════
// FILTER CONTROLS
// ══════════════════════════════════════════════════════════════
function initFilterControls() {
    // Signal filter chips
    $$('#signalFilterChips .chip').forEach(chip => {
        chip.addEventListener('click', () => {
            $$('#signalFilterChips .chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            renderResults();
        });
    });

    // Search
    let searchTimeout;
    $('#searchInput').addEventListener('input', () => {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(renderResults, 200);
    });

    // Timeframe filter
    $('#tfFilter').addEventListener('change', renderResults);

    // Column sorting
    $$('th.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const colMap = {
                name: 'Crypto Name',
                timeframe: 'Timeperiod',
                signal: 'Signal',
                angle: 'Angle',
                change: 'Daily Change'
            };
            const col = colMap[th.dataset.col];
            if (currentSort.col === col) {
                currentSort.asc = !currentSort.asc;
            } else {
                currentSort.col = col;
                currentSort.asc = true;
            }

            // Update sort icons
            $$('th.sortable').forEach(t => {
                t.classList.remove('sorted-asc', 'sorted-desc');
            });
            th.classList.add(currentSort.asc ? 'sorted-asc' : 'sorted-desc');
            renderResults();
        });
    });
}

// ══════════════════════════════════════════════════════════════
// CHART MODAL (TradingView Embed)
// ══════════════════════════════════════════════════════════════
function initChartModal() {
    $('#chartModalClose').addEventListener('click', closeChart);
    $('#chartModal').addEventListener('click', (e) => {
        if (e.target === $('#chartModal')) closeChart();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeChart();
    });
}

function openChart(indexName, timeframe) {
    // Map index names to TradingView symbols
    const tvSymbols = {
        'NIFTY': 'NSE:NIFTY',
        'BANKNIFTY': 'NSE:BANKNIFTY',
        'DOW JONES': 'FOREXCOM:DJI',
        'NASDAQ': 'NASDAQ:NDX'
    };

    const tvTimeframes = {
        '15min': '15', '30min': '30', '45min': '45',
        '1hr': '60', '2hr': '120', '4hr': '240',
        '1 day': 'D', '1 week': 'W'
    };

    const symbol = tvSymbols[indexName] || `NASDAQ:${indexName}`;
    const interval = tvTimeframes[timeframe] || '60';

    $('#chartModalTitle').textContent = `${indexName} — ${timeframe}`;

    // TradingView Advanced Chart Widget
    const container = $('#tradingviewWidget');
    container.innerHTML = `
        <iframe
            src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_widget&symbol=${encodeURIComponent(symbol)}&interval=${interval}&hidesidetoolbar=0&symboledit=1&saveimage=1&toolbarbg=0B1120&studies=[]&theme=dark&style=1&timezone=exchange&withdateranges=1&showpopupbutton=1&studies_overrides={}&overrides={}&enabled_features=[]&disabled_features=[]&showpopupbutton=1&locale=en"
            style="width:100%;height:100%;border:none;"
            allowtransparency="true"
            allowfullscreen>
        </iframe>
    `;

    $('#chartModal').classList.add('active');
}

function closeChart() {
    $('#chartModal').classList.remove('active');
    $('#tradingviewWidget').innerHTML = '';
}

// ══════════════════════════════════════════════════════════════
// MOBILE MENU
// ══════════════════════════════════════════════════════════════
function initMobileMenu() {
    const btn = $('#hamburgerBtn');
    if (btn) {
        btn.addEventListener('click', () => {
            $('#sidebar').classList.toggle('open');
        });
    }
}

// ══════════════════════════════════════════════════════════════
// EXPORT CSV
// ══════════════════════════════════════════════════════════════
function exportCSV() {
    if (allResults.length === 0) {
        showToast('No data to export', 'warning');
        return;
    }
    const headers = ['Index', 'Timeframe', 'Signal', 'Angle', 'Daily Change', 'Timestamp'];
    const rows = allResults.map(r => [
        r['Crypto Name'], r.Timeperiod, r.Signal, r.Angle, r['Daily Change'], r.Timestamp
    ]);
    const csv = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `gemini_scan_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    showToast('CSV exported', 'success');
}

// ══════════════════════════════════════════════════════════════
// CONNECTION STATUS
// ══════════════════════════════════════════════════════════════
function setConnectionStatus(online) {
    const statusEl = $('#connectionStatus');
    const dot = statusEl.querySelector('.status-dot');
    const text = statusEl.querySelector('span');
    dot.className = `status-dot ${online ? 'online' : 'offline'}`;
    text.textContent = online ? 'Connected' : 'Disconnected';
}

// ══════════════════════════════════════════════════════════════
// TOAST NOTIFICATIONS
// ══════════════════════════════════════════════════════════════
function showToast(message, type = 'info') {
    const container = $('#toastContainer');
    const iconMap = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<i class="fas ${iconMap[type]}"></i><span>${message}</span>`;
    container.appendChild(toast);

    requestAnimationFrame(() => {
        requestAnimationFrame(() => toast.classList.add('show'));
    });

    setTimeout(() => {
        toast.classList.remove('show');
        setTimeout(() => toast.remove(), 400);
    }, 4000);
}
