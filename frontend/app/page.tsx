"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { TrendingUp, Target, BarChart3, Loader2 } from "lucide-react"
import Plot from 'react-plotly.js'

const INTERVALS = ["1h", "4h"]
const PERIOD_MAP = { "1h": "15d", "4h": "30d" }

function isWithinMarketRefreshWindow() {
  const easternNow = new Date(new Date().toLocaleString('en-US', { timeZone: 'America/New_York' }))
  const mins = easternNow.getHours() * 60 + easternNow.getMinutes()
  return mins >= 565 && mins <= 965 // 9:25am–4:05pm ET
}

export default function MurliTradingApp() {
  const [tickerSymbol, setTickerSymbol] = useState("SPY")
  const [activeTimeframe, setActiveTimeframe] = useState("1h")
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
    } catch {
      setBackendData(null)
    } finally {
      if (triggeredByUser) setIsUserLoading(false)
      else setIsAutoRefreshing(false)
    }
  }, [])

  useEffect(() => {
    fetchAll(false)
    const id = setInterval(() => { if (isWithinMarketRefreshWindow()) fetchAll(false) }, 60000)
    return () => clearInterval(id)
  }, [fetchAll])

  const handleTickerChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setTickerSymbol(e.target.value)
    tickerRef.current = e.target.value
  }

  const consensus = backendData?.consensus
  const intervals = backendData?.intervals || {}
  const activeData = intervals[activeTimeframe] || {}

  const rrColor = (rr: number | null | undefined) => {
    if (rr == null) return "text-neutral-400"
    if (rr >= 2) return "text-emerald-400"
    if (rr >= 1) return "text-yellow-400"
    return "text-rose-400"
  }

  return (
    <div className="min-h-screen bg-black text-white">
      <div className="p-6 space-y-6">

        {/* Header */}
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-rose-500 rounded flex items-center justify-center">
              <BarChart3 className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-3xl font-bold">MURLI by sm0h</h1>
          </div>
          <div className="flex items-center space-x-3">
            <label className="text-sm text-neutral-300">Ticker:</label>
            <Input
              value={tickerSymbol}
              onChange={handleTickerChange}
              className="bg-neutral-800 border-neutral-600 text-white w-24"
              placeholder="SPY"
            />
            <Button
              className="bg-purple-600 hover:bg-purple-700"
              onClick={() => fetchAll(true)}
              disabled={isUserLoading}
            >
              <TrendingUp className="w-4 h-4 mr-2" />
              {isUserLoading ? "Loading…" : "Run Prediction"}
            </Button>
            {isAutoRefreshing && <Loader2 className="animate-spin text-neutral-400" size={18} />}
          </div>
        </div>

        {/* Signal Card */}
        <Card className="bg-neutral-900 border-neutral-700">
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <Target className="w-4 h-4 text-pink-500" />
              <span>Signal</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            {!consensus ? (
              <p className="text-neutral-400">Run prediction to see a signal.</p>
            ) : consensus.is_trade ? (
              <div className={`rounded-lg border-2 p-5 ${consensus.signal === "CALL"
                ? "border-emerald-500 bg-emerald-950/30"
                : "border-rose-500 bg-rose-950/30"}`}>
                <div className="text-2xl font-bold mb-3">
                  {consensus.signal === "CALL" ? "📈 BUY CALL" : "📉 BUY PUT"}
                  {" — "}
                  {tickerRef.current.toUpperCase()} ${consensus.strike} Strike
                </div>
                <div className="text-sm text-neutral-300 mb-4">
                  {consensus.signal === "CALL" ? "Expecting price to rise" : "Expecting price to fall"} toward ${consensus.tp?.toFixed(2)}
                  {" · "}{consensus.agreeing} timeframes agree
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="text-neutral-400 mb-1">Entry</div>
                    <div className="text-xl font-mono font-semibold">${consensus.avg_entry?.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-neutral-400 mb-1">Strike</div>
                    <div className="text-xl font-mono font-semibold">${consensus.strike}</div>
                  </div>
                  <div>
                    <div className="text-neutral-400 mb-1">Stop Loss</div>
                    <div className="text-xl font-mono font-semibold text-rose-400">${consensus.sl?.toFixed(2)}</div>
                  </div>
                  <div>
                    <div className="text-neutral-400 mb-1">Take Profit</div>
                    <div className="text-xl font-mono font-semibold text-emerald-400">${consensus.tp?.toFixed(2)}</div>
                  </div>
                </div>
                <div className={`mt-3 text-sm font-semibold ${rrColor(consensus.risk_reward)}`}>
                  R/R Ratio: {consensus.risk_reward != null ? consensus.risk_reward.toFixed(2) : "—"}
                </div>
              </div>
            ) : (
              <div className="rounded-lg border-2 border-yellow-600 bg-yellow-950/20 p-5">
                <div className="text-2xl font-bold mb-2">⚠️ No Trade</div>
                <div className="text-neutral-300 text-sm">{consensus.reason || "Mixed signals — wait for timeframes to align."}</div>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Timeframe tabs */}
        <Tabs value={activeTimeframe} onValueChange={setActiveTimeframe}>
          <TabsList className="bg-neutral-800">
            {INTERVALS.map((ivl) => (
              <TabsTrigger key={ivl} value={ivl} className="data-[state=active]:bg-rose-500">
                {ivl}
              </TabsTrigger>
            ))}
          </TabsList>
        </Tabs>

        {/* Per-interval detail */}
        <Card className="bg-neutral-900 border-neutral-700">
          <CardHeader>
            <div className="flex items-center justify-between w-full">
              <CardTitle>{activeTimeframe} — {tickerRef.current.toUpperCase()}</CardTitle>
              <div className="text-xs text-neutral-400 space-x-4">
                <span>ATR: <span className={activeData.low_volatility ? "text-yellow-300" : "text-white"}>
                  {activeData.atr != null ? activeData.atr.toFixed(3) : "—"}
                </span></span>
                <span>ATR/Close: <span className="text-white">
                  {activeData.atr_ratio != null ? (activeData.atr_ratio * 100).toFixed(2) + "%" : "—"}
                </span></span>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            {activeData.prediction && activeData.prediction.predicted_type_name !== "Unknown" ? (
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4 text-sm">
                <div>
                  <div className="text-neutral-400 mb-1">Signal</div>
                  <div className="text-lg font-bold flex items-center gap-1">
                    {activeData.prediction.predicted_type_name}
                    {activeData.macd_tick && (
                      <span className="text-emerald-400" title="MACD confirms">✓</span>
                    )}
                  </div>
                </div>
                <div>
                  <div className="text-neutral-400 mb-1">Confidence</div>
                  <div className="text-lg font-bold text-emerald-400">
                    {(activeData.prediction.confidence * 100).toFixed(0)}%
                  </div>
                </div>
                <div>
                  <div className="text-neutral-400 mb-1">Target Price</div>
                  <div className="text-lg font-bold">${activeData.prediction.estimated_value?.toFixed(2)}</div>
                </div>
                <div>
                  <div className="text-neutral-400 mb-1">Projected (ET)</div>
                  <div className="text-lg font-bold">{activeData.projected_time_et || "—"}</div>
                </div>
              </div>
            ) : (
              <p className="text-neutral-400 mb-4 text-sm">No valid signal for {activeTimeframe} — showing chart only.</p>
            )}

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm mb-4">
              {[
                ["Support", activeData.support],
                ["Resistance", activeData.resistance],
                ["Entry", activeData.entry],
                ["Exit Target", activeData.exit_target],
              ].map(([label, val]) => (
                <div key={label as string}>
                  <span className="text-neutral-400">{label}: </span>
                  <span className="font-mono">{val != null ? `$${(val as number).toFixed(2)}` : "—"}</span>
                </div>
              ))}
              <div>
                <span className="text-neutral-400">R/R: </span>
                <span className={`font-mono ${rrColor(activeData.risk_reward)}`}>
                  {activeData.risk_reward != null ? activeData.risk_reward.toFixed(2) : "—"}
                </span>
              </div>
              <div>
                <span className="text-neutral-400">Valid Setup: </span>
                <span className={activeData.is_valid ? "text-emerald-400" : "text-rose-400"}>
                  {activeData.is_valid ? "Yes" : "No"}
                </span>
              </div>
              <div>
                <span className="text-neutral-400">Price: </span>
                <span className="font-mono">{activeData.latest_price != null ? `$${activeData.latest_price.toFixed(2)}` : "—"}</span>
              </div>
              <div>
                <span className="text-neutral-400">As of: </span>
                <span className="font-mono text-xs">{activeData.latest_timestamp || "—"}</span>
              </div>
            </div>

            {/* Chart */}
            <div style={{ width: "100%", height: 380 }}>
              <Plot
                data={[
                  {
                    x: (activeData.ohlcv || []).map((d: any) => d.Date),
                    y: (activeData.ohlcv || []).map((d: any) => d.Close),
                    type: "scatter",
                    mode: "lines",
                    name: "Close",
                    line: { color: "#818cf8", width: 2 },
                  },
                  ...(activeData.peaks?.length ? [{
                    x: activeData.peaks.map((i: number) => activeData.ohlcv[i]?.Date),
                    y: activeData.peaks.map((i: number) => activeData.ohlcv[i]?.Close),
                    type: "scatter",
                    mode: "markers",
                    name: "Peaks",
                    marker: { color: "#22c55e", size: 11, symbol: "triangle-up" },
                  }] : []),
                  ...(activeData.troughs?.length ? [{
                    x: activeData.troughs.map((i: number) => activeData.ohlcv[i]?.Date),
                    y: activeData.troughs.map((i: number) => activeData.ohlcv[i]?.Close),
                    type: "scatter",
                    mode: "markers",
                    name: "Troughs",
                    marker: { color: "#ef4444", size: 11, symbol: "triangle-down" },
                  }] : []),
                  ...(activeData.prediction?.estimated_value && activeData.ohlcv?.length ? [{
                    x: [activeData.ohlcv[activeData.ohlcv.length - 1]?.Date],
                    y: [activeData.prediction.estimated_value],
                    type: "scatter",
                    mode: "markers",
                    name: `Next ${activeData.prediction.predicted_type_name}`,
                    marker: { color: "#a78bfa", size: 16, symbol: "star", line: { color: "#fff", width: 1.5 } },
                  }] : []),
                  ...(activeData.support != null ? [{
                    x: (activeData.ohlcv || []).map((d: any) => d.Date),
                    y: Array((activeData.ohlcv || []).length).fill(activeData.support),
                    type: "scatter", mode: "lines", name: "Support",
                    line: { color: "#3b82f6", width: 1.5, dash: "dot" }, hoverinfo: "skip",
                  }] : []),
                  ...(activeData.resistance != null ? [{
                    x: (activeData.ohlcv || []).map((d: any) => d.Date),
                    y: Array((activeData.ohlcv || []).length).fill(activeData.resistance),
                    type: "scatter", mode: "lines", name: "Resistance",
                    line: { color: "#f87171", width: 1.5, dash: "dot" }, hoverinfo: "skip",
                  }] : []),
                  ...(activeData.entry != null ? [{
                    x: (activeData.ohlcv || []).map((d: any) => d.Date),
                    y: Array((activeData.ohlcv || []).length).fill(activeData.entry),
                    type: "scatter", mode: "lines", name: "Entry",
                    line: { color: "#c084fc", width: 1.5, dash: "dash" }, hoverinfo: "skip",
                  }] : []),
                ]}
                layout={{
                  autosize: true,
                  height: 380,
                  margin: { t: 20, r: 20, l: 50, b: 50 },
                  paper_bgcolor: "#18181b",
                  plot_bgcolor: "#18181b",
                  font: { color: "#fff", size: 11 },
                  xaxis: { title: "Date", tickformat: "%m/%d", showgrid: true, gridcolor: "#333", type: "date" },
                  yaxis: { title: "Price", showgrid: true, gridcolor: "#333" },
                  legend: { orientation: "h", y: -0.18, font: { size: 10 } },
                }}
                useResizeHandler={true}
                style={{ width: "100%", height: "100%" }}
                config={{ displayModeBar: false }}
              />
            </div>
          </CardContent>
        </Card>

      </div>
    </div>
  )
}
