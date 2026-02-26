"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { TrendingUp, Target, DollarSign, Clock, BarChart3, Loader2 } from "lucide-react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ResponsiveContainer, ReferenceLine, Scatter, Tooltip, Legend } from "recharts"
import Plot from 'react-plotly.js';

const INTERVALS = ["5m", "15m", "1h"];
const PERIOD_MAP = { "5m": "30d", "15m": "30d", "1h": "60d" };

function formatDateLikeOHLCV(date: Date) {
  return date.getFullYear() + '-' +
    String(date.getMonth() + 1).padStart(2, '0') + '-' +
    String(date.getDate()).padStart(2, '0') + ' ' +
    String(date.getHours()).padStart(2, '0') + ':' +
    String(date.getMinutes()).padStart(2, '0') + ':' +
    String(date.getSeconds()).padStart(2, '0');
}

function getNextMarketTime(last: Date, offsetMinutes: number) {
  // US market hours: 9:30am to 4:00pm
  const MARKET_OPEN_HOUR = 9;
  const MARKET_OPEN_MIN = 30;
  const MARKET_CLOSE_HOUR = 16;
  const MARKET_CLOSE_MIN = 0;
  let next = new Date(last.getTime() + offsetMinutes * 60 * 1000);
  // Check if next is after market close
  if (
    next.getHours() > MARKET_CLOSE_HOUR ||
    (next.getHours() === MARKET_CLOSE_HOUR && next.getMinutes() > MARKET_CLOSE_MIN)
  ) {
    // Move to next day 9:30am
    next.setDate(next.getDate() + 1);
    next.setHours(MARKET_OPEN_HOUR, MARKET_OPEN_MIN, 0, 0);
  }
  // If before market open, set to 9:30am
  if (
    next.getHours() < MARKET_OPEN_HOUR ||
    (next.getHours() === MARKET_OPEN_HOUR && next.getMinutes() < MARKET_OPEN_MIN)
  ) {
    next.setHours(MARKET_OPEN_HOUR, MARKET_OPEN_MIN, 0, 0);
  }
  return next;
}

// Helper to check if now is between 9:25am and 4:05pm US Eastern
function isWithinMarketRefreshWindow() {
  // Get current time in UTC
  const now = new Date();
  // Convert to US Eastern Time (handles DST)
  const easternOffset = -5 * 60; // EST is UTC-5, but this does not handle DST
  // Use Intl API for robust conversion
  const easternNow = new Date(now.toLocaleString('en-US', { timeZone: 'America/New_York' }));
  const hour = easternNow.getHours();
  const min = easternNow.getMinutes();
  // 9:25am = 9*60+25 = 565, 4:05pm = 16*60+5 = 965
  const mins = hour * 60 + min;
  return mins >= 565 && mins <= 965;
}

export default function MurliTradingApp() {
  const [tickerSymbol, setTickerSymbol] = useState("SPY")
  const [activeTimeframe, setActiveTimeframe] = useState("15m")
  const [backendData, setBackendData] = useState<any>(null)
  const [isUserLoading, setIsUserLoading] = useState(false)
  const [isAutoRefreshing, setIsAutoRefreshing] = useState(false)
  const tickerRef = useRef(tickerSymbol)

  const fetchAll = useCallback(async (triggeredByUser = false) => {
    if (triggeredByUser) setIsUserLoading(true)
    else setIsAutoRefreshing(true)
    try {
      const res = await fetch("http://127.0.0.1:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: tickerRef.current, intervals: INTERVALS, period_map: PERIOD_MAP }),
      })
      const data = await res.json()
      setBackendData(data)
    } catch (err) {
      setBackendData(null)
    } finally {
      if (triggeredByUser) setIsUserLoading(false)
      else setIsAutoRefreshing(false)
    }
  }, [])

  // Polling every 10 seconds (background, no spinner)
  useEffect(() => {
    fetchAll(false)
    const interval = setInterval(() => {
      if (isWithinMarketRefreshWindow()) {
        fetchAll(false)
      }
    }, 10000)
    return () => clearInterval(interval)
  }, [fetchAll])

  // Handler for manual refresh
  const handleManualRefresh = () => {
    fetchAll(true)
  }

  // Handler for ticker change
  const handleTickerChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTickerSymbol(e.target.value)
    tickerRef.current = e.target.value
    fetchAll(true)
  }

  const consensus = backendData?.consensus
  const intervals = backendData?.intervals || {}
  const activeData = intervals[activeTimeframe] || {}

  // Debug: log ohlcv, peaks, and troughs for all intervals
  if (typeof window !== 'undefined' && backendData?.intervals) {
    ['5m', '15m', '1h'].forEach(ivl => {
      const d = backendData.intervals[ivl] || {};
      console.log(`${ivl} ohlcv length:`, (d.ohlcv || []).length);
      console.log(`${ivl} ohlcv sample:`, (d.ohlcv || []).slice(0, 3));
      console.log(`${ivl} peaks:`, d.peaks);
      console.log(`${ivl} troughs:`, d.troughs);
    });
  }
  // Debug: log last 5 Close prices for the active interval
  if (typeof window !== 'undefined' && activeData.ohlcv) {
    console.log('Last 5 Close prices:', activeData.ohlcv.slice(-5).map((d: any) => d.Close));
  }

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="p-6 space-y-6">
        {/* Header with Ticker Input */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-rose-500 rounded flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-3xl font-bold">MURLI by sm0h</h1>
          </div>

          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <label className="text-sm text-neutral-300">Ticker Symbol:</label>
              <Input
                value={tickerSymbol}
                onChange={handleTickerChange}
                className="bg-neutral-800 border-neutral-600 text-white w-24"
                placeholder="SPY"
              />
            </div>
            <Button className="bg-purple-600 hover:bg-purple-700" onClick={handleManualRefresh} disabled={isUserLoading}>
              <TrendingUp className="w-4 h-4 mr-2" />
              {isUserLoading ? "Loading..." : "Run Pivot Prediction"}
            </Button>
          </div>
        </div>

        {/* Consensus Summary */}
        <Card className="bg-neutral-900 border-neutral-700">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Target className="w-4 h-4 text-pink-500" />
              <span>Consensus</span>
              {isAutoRefreshing && (
                <Loader2 className="animate-spin text-neutral-400 ml-2" size={20} />
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {consensus ? (
              <div className="space-y-2">
                <div className="text-lg font-bold">{consensus.type}</div>
                {consensus.avg_entry !== undefined && (
                  <div className="flex space-x-6 text-sm">
                    <span>Avg Entry: <span className="text-emerald-400 font-mono">${consensus.avg_entry?.toFixed(2)}</span></span>
                    <span>Avg Exit: <span className="text-emerald-400 font-mono">${consensus.avg_exit?.toFixed(2)}</span></span>
                    <span>Risk/Reward: <span className="text-rose-400 font-mono">{consensus.risk_reward?.toFixed(2)}</span></span>
                  </div>
                )}
              </div>
            ) : (
              <div>No consensus data.</div>
            )}
          </CardContent>
        </Card>

        {/* Interval Tabs */}
        <Tabs value={activeTimeframe} onValueChange={setActiveTimeframe}>
          <TabsList className="bg-neutral-800">
            {INTERVALS.map((ivl) => (
              <TabsTrigger key={ivl} value={ivl} className="data-[state=active]:bg-rose-500">
                {ivl}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>

        {/* Interval Details */}
        <Card className="bg-neutral-900 border-neutral-700">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Clock className="w-5 h-5 text-neutral-400" />
              <span>{activeTimeframe} Pivot Prediction</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {activeData.prediction ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <div className="text-sm text-neutral-400 mb-1">Pivot Type</div>
                  <div className="text-2xl font-bold flex items-center">
                    {activeData.prediction.predicted_type_name}
                    {activeData.macd_tick && (
                      <span className="text-white font-bold ml-2" title="MACD agrees">
                        <svg xmlns="http://www.w3.org/2000/svg" className="inline w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" /></svg>
                      </span>
                    )}
                  </div>
                </div>
                <div>
                  <div className="text-sm text-neutral-400 mb-1">Confidence</div>
                  <div className="text-2xl font-bold text-emerald-400">{(activeData.prediction.confidence * 100).toFixed(1)}%</div>
                </div>
                <div>
                  <div className="text-sm text-neutral-400 mb-1">Target Price</div>
                  <div className="text-2xl font-bold">${activeData.prediction.estimated_value?.toFixed(2)}</div>
                </div>
                <div>
                  <div className="text-sm text-neutral-400 mb-1">Projected Time (ET)</div>
                  <div className="text-2xl font-bold">{activeData.projected_time_et || '-'}</div>
                </div>
              </div>
            ) : (
              <div>No prediction data available.</div>
            )}
            <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-neutral-400">Entry:</span> <span className="font-mono">${activeData.entry?.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-neutral-400">Exit Target:</span> <span className="font-mono">${activeData.exit_target?.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-neutral-400">Support:</span> <span className="font-mono">${activeData.support?.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-neutral-400">Resistance:</span> <span className="font-mono">${activeData.resistance?.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-neutral-400">Risk/Reward:</span> <span className="font-mono">{activeData.risk_reward !== undefined ? activeData.risk_reward?.toFixed(2) : "-"}</span>
              </div>
              <div>
                <span className="text-neutral-400">Valid Setup:</span> <span className={activeData.is_valid ? "text-emerald-400" : "text-rose-400"}>{activeData.is_valid ? "Yes" : "No"}</span>
              </div>
              <div>
                <span className="text-neutral-400">Latest Price:</span> <span className="font-mono">${activeData.latest_price?.toFixed(2)}</span>
              </div>
              <div>
                <span className="text-neutral-400">Timestamp:</span> <span className="font-mono">{activeData.latest_timestamp}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Chart Section */}
        <Card className="bg-neutral-900 border-neutral-700">
          <CardHeader>
            <div className="flex items-center justify-between w-full">
              <CardTitle>Price & Wave Chart ({activeTimeframe})</CardTitle>
              <div className="flex flex-col items-end text-xs text-neutral-300 space-y-0.5">
                <span title="Average True Range (ATR)">
                  <span className={activeData.low_volatility ? 'font-mono text-yellow-300 flex items-center' : 'font-mono text-white flex items-center'}>
                    ATR: {activeData.atr !== undefined && activeData.atr !== null ? activeData.atr.toFixed(3) : '-'}
                    {activeData.low_volatility && <span className="ml-1" title="Low Volatility"><svg xmlns="http://www.w3.org/2000/svg" className="inline w-3 h-3 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01M21 12c0 4.97-4.03 9-9 9s-9-4.03-9-9 4.03-9 9-9 9 4.03 9 9z" /></svg></span>}
                  </span>
                </span>
                <span title="ATR / Close (Volatility Ratio)">
                  ATR/Close: <span className="font-mono text-white">{activeData.atr_ratio !== undefined && activeData.atr_ratio !== null ? (activeData.atr_ratio * 100).toFixed(2) + '%' : '-'}</span>
                </span>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div style={{ width: '100%', height: 350 }}>
              <Plot
                data={[
                  // Close price line
                  {
                    x: (activeData.ohlcv || []).map((d: any) => d.Date),
                    y: (activeData.ohlcv || []).map((d: any) => d.Close),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Close Price',
                    line: { color: '#8884d8', width: 2.5 },
                    hovertemplate: '<span style="color:#fff;background:#222;padding:2px 6px;border-radius:4px"><b>$%{y:.2f}</b></span><extra></extra>',
                  },
                  // Peaks (green upward triangles)
                  ...(activeData.peaks && activeData.peaks.length > 0 ? [
                    {
                      x: activeData.peaks.map((idx: number) => activeData.ohlcv[idx]?.Date),
                      y: activeData.peaks.map((idx: number) => activeData.ohlcv[idx]?.Close),
                      type: 'scatter',
                      mode: 'markers',
                      name: 'Peaks',
                      marker: { color: '#2ca02c', size: 12, symbol: 'triangle-up' },
                      hovertemplate: '<span style="color:#fff;background:#22c55e;padding:2px 6px;border-radius:4px"><b>$%{y:.2f}</b></span><extra></extra>',
                      showlegend: true,
                    }
                  ] : []),
                  // Troughs (red downward triangles)
                  ...(activeData.troughs && activeData.troughs.length > 0 ? [
                    {
                      x: activeData.troughs.map((idx: number) => activeData.ohlcv[idx]?.Date),
                      y: activeData.troughs.map((idx: number) => activeData.ohlcv[idx]?.Close),
                      type: 'scatter',
                      mode: 'markers',
                      name: 'Troughs',
                      marker: { color: '#d62728', size: 12, symbol: 'triangle-down' },
                      hovertemplate: '<span style="color:#fff;background:#ef4444;padding:2px 6px;border-radius:4px"><b>$%{y:.2f}</b></span><extra></extra>',
                      showlegend: true,
                    }
                  ] : []),
                  // Predicted next pivot dot (purple)
                  ...((activeData.prediction && activeData.ohlcv && activeData.ohlcv.length > 0) ? (() => {
                    const lastDateStr = activeData.ohlcv?.[activeData.ohlcv.length - 1]?.Date;
                    let xValue = lastDateStr;
                    if (lastDateStr) {
                      // Parse as local time
                      const last = new Date(lastDateStr.replace(' ', 'T'));
                      let next;
                      if (activeTimeframe === '5m' || activeTimeframe === '15m') {
                        next = getNextMarketTime(last, 60); // 1 hour
                      } else if (activeTimeframe === '1h') {
                        next = getNextMarketTime(last, 900); // 15 hours
                      } else {
                        next = last;
                      }
                      xValue = formatDateLikeOHLCV(next);
                      console.log('OHLCV Date:', lastDateStr, 'Dot xValue:', xValue);
                    }
                    return [{
                      x: [xValue],
                      y: [activeData.prediction.estimated_value],
                      type: 'scatter',
                      mode: 'markers',
                      name: 'Next Pivot',
                      marker: { color: '#a259e6', size: 18, symbol: 'circle', line: { color: '#fff', width: 2 } },
                      hovertemplate: `${activeData.prediction.predicted_type_name} (Conf: ${(activeData.prediction.confidence * 100).toFixed(1)}%)<br>Price: $${activeData.prediction.estimated_value?.toFixed(2)}`,
                      showlegend: true,
                    }];
                  })() : []),
                  // Support line
                  ...(activeData.support !== undefined ? [{
                    x: (activeData.ohlcv || []).map((d: any) => d.Date),
                    y: Array((activeData.ohlcv || []).length).fill(activeData.support),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Support',
                    line: { color: '#3b82f6', width: 2, dash: 'dot' },
                    hoverinfo: 'skip',
                    showlegend: true,
                  }] : []),
                  // Entry line
                  ...(activeData.entry !== undefined ? [{
                    x: (activeData.ohlcv || []).map((d: any) => d.Date),
                    y: Array((activeData.ohlcv || []).length).fill(activeData.entry),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Entry',
                    line: { color: '#a259e6', width: 2, dash: 'dash' },
                    hoverinfo: 'skip',
                    showlegend: true,
                  }] : []),
                  // Resistance line
                  ...(activeData.resistance !== undefined ? [{
                    x: (activeData.ohlcv || []).map((d: any) => d.Date),
                    y: Array((activeData.ohlcv || []).length).fill(activeData.resistance),
                    type: 'scatter',
                    mode: 'lines',
                    name: 'Resistance',
                    line: { color: '#ef4444', width: 2, dash: 'dot' },
                    hoverinfo: 'skip',
                    showlegend: true,
                  }] : []),
                ]}
                layout={{
                  autosize: true,
                  height: 350,
                  margin: { t: 30, r: 30, l: 40, b: 40 },
                  paper_bgcolor: '#18181b',
                  plot_bgcolor: '#18181b',
                  font: { color: '#fff' },
                  hoverdistance: 5,
                  spikedistance: 5,
                  xaxis: {
                    title: 'Date',
                    tickformat: '%m-%d',
                    showgrid: true,
                    gridcolor: '#444',
                    type: 'date',
                  },
                  yaxis: {
                    title: 'Price',
                    showgrid: true,
                    gridcolor: '#444',
                  },
                  legend: { orientation: 'h', y: -0.2 },
                }}
                useResizeHandler={true}
                style={{ width: '100%', height: '100%' }}
                config={{ displayModeBar: false }}
              />
            </div>
          </CardContent>
        </Card>

        {/* SPY–QQQ Convergence/Divergence Box (15m only) */}
        {backendData?.intervals?.["15m"] && (
          <Card className="bg-neutral-900 border-neutral-700">
            <CardHeader>
              <CardTitle className="flex items-center space-x-2">
                <BarChart3 className="w-4 h-4 text-yellow-400" />
                <span>SPY–QQQ Convergence/Divergence (15m)</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              {backendData.intervals["15m"].spy_pct_change !== undefined && backendData.intervals["15m"].qqq_pct_change !== undefined ? (
                <div className="flex flex-col md:flex-row md:items-center md:space-x-8 space-y-2 md:space-y-0 text-sm">
                  <span>
                    <span className="text-neutral-400">SPY % Change (3 bars): </span>
                    <span className={backendData.intervals["15m"].spy_pct_change > 0 ? "text-emerald-400" : backendData.intervals["15m"].spy_pct_change < 0 ? "text-rose-400" : "text-white"}>
                      {(backendData.intervals["15m"].spy_pct_change * 100).toFixed(2)}%
                    </span>
                  </span>
                  <span>
                    <span className="text-neutral-400">QQQ % Change (3 bars): </span>
                    <span className={backendData.intervals["15m"].qqq_pct_change > 0 ? "text-emerald-400" : backendData.intervals["15m"].qqq_pct_change < 0 ? "text-rose-400" : "text-white"}>
                      {(backendData.intervals["15m"].qqq_pct_change * 100).toFixed(2)}%
                    </span>
                  </span>
                  <span>
                    <span className="text-neutral-400">Status: </span>
                    {backendData.intervals["15m"].convergence_status === "convergent" && <span className="text-emerald-400 font-bold">Convergent &#x2714;</span>}
                    {backendData.intervals["15m"].convergence_status === "divergent" && <span className="text-yellow-300 font-bold">Divergent &#9888;</span>}
                    {backendData.intervals["15m"].convergence_status === "neutral" && <span className="text-white">Neutral</span>}
                  </span>
                  {backendData.intervals["15m"].leader && (
                    <span>
                      <span className="text-neutral-400">Leader: </span>
                      <span className="font-bold text-white">{backendData.intervals["15m"].leader}</span>
                      <span className={backendData.intervals["15m"].leader_direction === "up" ? "text-emerald-400 ml-1" : backendData.intervals["15m"].leader_direction === "down" ? "text-rose-400 ml-1" : "ml-1"}>
                        {backendData.intervals["15m"].leader_direction === "up" ? <>&uarr;</> : backendData.intervals["15m"].leader_direction === "down" ? <>&darr;</> : null}
                      </span>
                    </span>
                  )}
                  {backendData.intervals["15m"].signal_agrees_with_leader !== null && backendData.intervals["15m"].signal_agrees_with_leader !== undefined && (
                    <span>
                      <span className="text-neutral-400">Signal Agrees with Leader: </span>
                      {backendData.intervals["15m"].signal_agrees_with_leader ? (
                        <span className="text-emerald-400 font-bold">Yes &#x2714;</span>
                      ) : (
                        <span className="text-rose-400 font-bold">No &#10060;</span>
                      )}
                    </span>
                  )}
                </div>
              ) : (
                <div className="text-neutral-400">No SPY–QQQ convergence data available.</div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}
