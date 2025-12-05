import { IconTrendingUp, IconTrendingDown } from "@tabler/icons-react"

import { Badge } from "@/components/ui/badge"
import {
  Card,
  CardAction,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import type { PredictionEntry } from "@/lib/storage"

interface SectionCardsProps {
  lastPrediction: PredictionEntry | null
  predictionHistory: PredictionEntry[]
  getRiskBadgeVariant: (riskLabel: string) => 'default' | 'destructive' | 'outline'
  getRiskBadgeColor: (riskLabel: string) => string
}

export function SectionCards({ 
  lastPrediction, 
  predictionHistory,
  getRiskBadgeVariant,
  getRiskBadgeColor,
}: SectionCardsProps) {
  // Calculate average predicted glucose from recent predictions
  const recentPredictions = predictionHistory.slice(0, 10)
  const avgPredicted = recentPredictions.length > 0
    ? recentPredictions.reduce((sum, entry) => sum + entry.result.predicted_glucose_mgdl, 0) / recentPredictions.length
    : 0

  // Determine trend (simplified - compare last 2 predictions)
  const trend = predictionHistory.length >= 2
    ? predictionHistory[0].result.predicted_glucose_mgdl - predictionHistory[1].result.predicted_glucose_mgdl
    : 0

  const currentGlucose = lastPrediction?.input.current_glucose_mgdl ?? 0
  const predictedGlucose = lastPrediction?.result.predicted_glucose_mgdl ?? 0
  const riskLabel = lastPrediction?.result.risk_label ?? 'N/A'
  const totalPredictions = predictionHistory.length

  // Determine if trend is positive or negative (for glucose, lower is generally better, but it depends on context)
  const isPositiveTrend = trend <= 0 // Negative trend (decreasing) is good for high glucose

  return (
    <div className="*:data-[slot=card]:from-primary/5 *:data-[slot=card]:to-card dark:*:data-[slot=card]:bg-card grid grid-cols-1 gap-4 px-4 *:data-[slot=card]:bg-gradient-to-t *:data-[slot=card]:shadow-xs lg:px-6 @xl/main:grid-cols-2 @5xl/main:grid-cols-4">
      <Card className="@container/card">
        <CardHeader>
          <CardDescription>This Morning's Glucose Reading</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {currentGlucose > 0 ? `${currentGlucose.toFixed(0)}` : '—'} mg/dL
          </CardTitle>
          {lastPrediction && (
            <CardAction>
              <Badge variant={getRiskBadgeVariant(riskLabel)} className={getRiskBadgeColor(riskLabel) + ' text-white'}>
                {riskLabel}
              </Badge>
            </CardAction>
          )}
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          {lastPrediction ? (
            <>
              <div className="line-clamp-1 flex gap-2 font-medium">
                Latest reading {lastPrediction.input.current_glucose_mgdl} mg/dL
              </div>
              <div className="text-muted-foreground">
                From your most recent prediction
              </div>
            </>
          ) : (
            <div className="text-muted-foreground">
              Submit a prediction to see your glucose level
            </div>
          )}
        </CardFooter>
      </Card>
      
      <Card className="@container/card">
        <CardHeader>
          <CardDescription>Predicted Glucose</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {predictedGlucose > 0 ? `${predictedGlucose.toFixed(1)}` : '—'} mg/dL
          </CardTitle>
          {predictionHistory.length >= 2 && (
            <CardAction>
              <Badge variant="outline">
                {isPositiveTrend ? (
                  <>
                    <IconTrendingDown />
                    {Math.abs(trend).toFixed(1)}
                  </>
                ) : (
                  <>
                    <IconTrendingUp />
                    {trend.toFixed(1)}
                  </>
                )}
              </Badge>
            </CardAction>
          )}
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          {lastPrediction ? (
            <>
              <div className="line-clamp-1 flex gap-2 font-medium">
                Predicted level {lastPrediction.result.predicted_glucose_mgdl.toFixed(1)} mg/dL
              </div>
              <div className="text-muted-foreground">
                Based on your latest input
              </div>
            </>
          ) : (
            <div className="text-muted-foreground">
              Prediction will appear after submission
            </div>
          )}
        </CardFooter>
      </Card>
      
      <Card className="@container/card">
        <CardHeader>
          <CardDescription>Risk Level</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {lastPrediction ? riskLabel : '—'}
          </CardTitle>
          {lastPrediction && (
            <CardAction>
              <Badge variant={getRiskBadgeVariant(riskLabel)} className={getRiskBadgeColor(riskLabel) + ' text-white'}>
                {riskLabel}
              </Badge>
            </CardAction>
          )}
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          {lastPrediction ? (
            <>
              <div className="line-clamp-1 flex gap-2 font-medium">
                Current risk assessment
              </div>
              <div className="text-muted-foreground">
                {lastPrediction.result.explanation}
              </div>
            </>
          ) : (
            <div className="text-muted-foreground">
              Risk level will be shown after prediction
            </div>
          )}
        </CardFooter>
      </Card>
      
      <Card className="@container/card">
        <CardHeader>
          <CardDescription>Total Predictions</CardDescription>
          <CardTitle className="text-2xl font-semibold tabular-nums @[250px]/card:text-3xl">
            {totalPredictions}
          </CardTitle>
          {recentPredictions.length > 0 && avgPredicted > 0 && (
            <CardAction>
              <Badge variant="outline">
                Avg: {avgPredicted.toFixed(1)}
              </Badge>
            </CardAction>
          )}
        </CardHeader>
        <CardFooter className="flex-col items-start gap-1.5 text-sm">
          <div className="line-clamp-1 flex gap-2 font-medium">
            {totalPredictions > 0 ? (
              <>
                Tracking {totalPredictions} prediction{totalPredictions !== 1 ? 's' : ''}
                {recentPredictions.length > 0 && (
                  <>
                    {' '}
                    <IconTrendingUp className="size-4" />
                  </>
                )}
              </>
            ) : (
              'No predictions yet'
            )}
          </div>
          <div className="text-muted-foreground">
            {totalPredictions > 0 
              ? `Average predicted: ${avgPredicted.toFixed(1)} mg/dL`
              : 'Start tracking your glucose predictions'}
          </div>
        </CardFooter>
      </Card>
    </div>
  )
}
