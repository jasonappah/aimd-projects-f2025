"use client"

import * as React from "react"
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ReferenceLine } from "recharts"
import { format } from "date-fns"

import { useIsMobile } from "@/hooks/use-mobile"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from "@/components/ui/chart"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  ToggleGroup,
  ToggleGroupItem,
} from "@/components/ui/toggle-group"
import type { PredictionEntry } from "@/lib/storage"

interface ChartAreaInteractiveProps {
  predictionHistory: PredictionEntry[]
}

export function ChartAreaInteractive({ predictionHistory }: ChartAreaInteractiveProps) {
  const isMobile = useIsMobile()
  const [timeRange, setTimeRange] = React.useState("90d")

  React.useEffect(() => {
    if (isMobile) {
      setTimeRange("7d")
    }
  }, [isMobile])

  // Filter data based on time range
  const filteredData = React.useMemo(() => {
    if (predictionHistory.length === 0) return []

    const now = new Date()
    let daysToSubtract = 90
    if (timeRange === "30d") {
      daysToSubtract = 30
    } else if (timeRange === "7d") {
      daysToSubtract = 7
    }

    const startDate = new Date(now)
    startDate.setDate(startDate.getDate() - daysToSubtract)

    return predictionHistory
      .filter((entry) => {
        const entryDate = new Date(entry.timestamp)
        return entryDate >= startDate
      })
      .reverse() // Reverse to show chronological order (oldest to newest)
      .map((entry) => ({
        timestamp: entry.timestamp,
        date: format(new Date(entry.timestamp), 'MM/dd HH:mm'),
        current: entry.input.current_glucose_mgdl,
        predicted: entry.result.predicted_glucose_mgdl,
        risk: entry.result.risk_label,
      }))
  }, [predictionHistory, timeRange])

  const chartConfig = {
    current: {
      label: "This Morning's Glucose Reading",
      color: "hsl(var(--chart-1))",
    },
    predicted: {
      label: "Predicted Glucose",
      color: "hsl(var(--chart-2))",
    },
  } satisfies ChartConfig

  if (predictionHistory.length === 0) {
    return (
      <Card className="@container/card">
        <CardHeader>
          <CardTitle>Prediction History</CardTitle>
          <CardDescription>
            Track your current and predicted glucose levels over time
          </CardDescription>
        </CardHeader>
        <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
          <div className="aspect-auto h-[250px] w-full flex items-center justify-center text-muted-foreground">
            <p>No prediction history yet. Submit a prediction to see the chart.</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="@container/card">
      <CardHeader>
        <CardTitle>Prediction History</CardTitle>
        <CardDescription>
          <span className="hidden @[540px]/card:block">
            Track your current and predicted glucose levels over time
          </span>
          <span className="@[540px]/card:hidden">Glucose levels over time</span>
        </CardDescription>
        <CardAction>
          <ToggleGroup
            type="single"
            value={timeRange}
            onValueChange={setTimeRange}
            variant="outline"
            className="hidden *:data-[slot=toggle-group-item]:!px-4 @[767px]/card:flex"
          >
            <ToggleGroupItem value="90d">Last 3 months</ToggleGroupItem>
            <ToggleGroupItem value="30d">Last 30 days</ToggleGroupItem>
            <ToggleGroupItem value="7d">Last 7 days</ToggleGroupItem>
          </ToggleGroup>
          <Select value={timeRange} onValueChange={setTimeRange}>
            <SelectTrigger
              className="flex w-40 **:data-[slot=select-value]:block **:data-[slot=select-value]:truncate @[767px]/card:hidden"
              size="sm"
              aria-label="Select a time range"
            >
              <SelectValue placeholder="Last 3 months" />
            </SelectTrigger>
            <SelectContent className="rounded-xl">
              <SelectItem value="90d" className="rounded-lg">
                Last 3 months
              </SelectItem>
              <SelectItem value="30d" className="rounded-lg">
                Last 30 days
              </SelectItem>
              <SelectItem value="7d" className="rounded-lg">
                Last 7 days
              </SelectItem>
            </SelectContent>
          </Select>
        </CardAction>
      </CardHeader>
      <CardContent className="px-2 pt-4 sm:px-6 sm:pt-6">
        <ChartContainer
          config={chartConfig}
          className="aspect-auto h-[250px] w-full"
        >
          <LineChart data={filteredData}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="date"
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              minTickGap={32}
              angle={-45}
              textAnchor="end"
              height={80}
            />
            <YAxis
              tickLine={false}
              axisLine={false}
              tickMargin={8}
              label={{
                value: 'Glucose (mg/dL)',
                angle: -90,
                position: 'insideLeft',
              }}
            />
            <ChartTooltip
              cursor={false}
              content={
                <ChartTooltipContent
                  labelFormatter={(value) => {
                    // Value is the formatted date string from dataKey="date"
                    return value
                  }}
                  indicator="dot"
                />
              }
            />
            <ChartLegend content={<ChartLegendContent />} />
            <ReferenceLine
              y={70}
              stroke="#ef4444"
              strokeDasharray="3 3"
              label={{ value: 'Hypo Threshold', position: 'right' }}
            />
            <ReferenceLine
              y={180}
              stroke="#ef4444"
              strokeDasharray="3 3"
              label={{ value: 'Hyper Threshold', position: 'right' }}
            />
            <Line
              type="monotone"
              dataKey="current"
              stroke="hsl(var(--chart-1))"
              strokeWidth={2}
              dot={{ r: 4 }}
              name="current"
            />
            <Line
              type="monotone"
              dataKey="predicted"
              stroke="hsl(var(--chart-2))"
              strokeWidth={2}
              dot={{ r: 4 }}
              name="predicted"
            />
          </LineChart>
        </ChartContainer>
      </CardContent>
    </Card>
  )
}
