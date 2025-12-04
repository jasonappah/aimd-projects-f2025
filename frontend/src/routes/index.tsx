import { createFileRoute } from '@tanstack/react-router'
import { useState, useEffect } from 'react'

import { AppSidebar } from '@/components/app-sidebar'
import { ChartAreaInteractive } from '@/components/chart-area-interactive'
import { DataTable } from '@/components/data-table'
import { SectionCards } from '@/components/section-cards'
import { SiteHeader } from '@/components/site-header'
import { PredictionForm } from '@/components/prediction-form'
import {
  SidebarInset,
  SidebarProvider,
} from '@/components/ui/sidebar'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { format } from 'date-fns'
import type { PredictionEntry } from '@/lib/storage'
import { getPredictionEntries } from '@/lib/storage'

export const Route = createFileRoute('/')({
  component: App,
})

function App() {
  const [predictionHistory, setPredictionHistory] = useState<PredictionEntry[]>([])
  const [lastPrediction, setLastPrediction] = useState<PredictionEntry | null>(null)

  const refreshHistory = () => {
    const entries = getPredictionEntries()
    setPredictionHistory(entries)
    if (entries.length > 0) {
      setLastPrediction(entries[0])
    } else {
      setLastPrediction(null)
    }
  }

  // Load history from localStorage on mount
  useEffect(() => {
    refreshHistory()
  }, [])

  // Get risk badge variant
  const getRiskBadgeVariant = (riskLabel: string): 'default' | 'destructive' | 'outline' => {
    switch (riskLabel.toLowerCase()) {
      case 'hypo':
      case 'hyper':
        return 'destructive'
      case 'borderline':
        return 'outline'
      case 'normal':
        return 'default'
      default:
        return 'outline'
    }
  }

  // Get risk badge color class
  const getRiskBadgeColor = (riskLabel: string): string => {
    switch (riskLabel.toLowerCase()) {
      case 'hypo':
        return 'bg-red-500'
      case 'hyper':
        return 'bg-red-600'
      case 'borderline':
        return 'bg-yellow-500'
      case 'normal':
        return 'bg-green-500'
      default:
        return ''
    }
  }

  return (
    <SidebarProvider
      style={
        {
          '--sidebar-width': 'calc(var(--spacing) * 72)',
          '--header-height': 'calc(var(--spacing) * 12)',
        } as React.CSSProperties
      }
    >
      <AppSidebar variant="inset" />
      <SidebarInset>
        <SiteHeader />
        <div className="flex flex-1 flex-col">
          <div className="@container/main flex flex-1 flex-col gap-2">
            <div className="flex flex-col gap-4 py-4 md:gap-6 md:py-6">
              <SectionCards 
                lastPrediction={lastPrediction}
                predictionHistory={predictionHistory}
                getRiskBadgeVariant={getRiskBadgeVariant}
                getRiskBadgeColor={getRiskBadgeColor}
              />
              
              <div className="grid gap-4 px-4 lg:px-6 md:grid-cols-2">
                <PredictionForm onSuccess={refreshHistory} />
                
                {/* Latest Prediction Card */}
                <Card>
                  <CardHeader>
                    <CardTitle>Latest Prediction</CardTitle>
                    <CardDescription>Your most recent blood sugar prediction result</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {lastPrediction ? (
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <div>
                            <p className="text-sm text-muted-foreground">Predicted Glucose</p>
                            <p className="text-4xl font-bold">
                              {lastPrediction.result.predicted_glucose_mgdl.toFixed(1)} mg/dL
                            </p>
                          </div>
                          <Badge
                            variant={getRiskBadgeVariant(lastPrediction.result.risk_label)}
                            className={`${getRiskBadgeColor(lastPrediction.result.risk_label)} text-white`}
                          >
                            {lastPrediction.result.risk_label}
                          </Badge>
                        </div>

                        <div className="pt-4 border-t">
                          <p className="text-sm font-medium mb-2">Current Glucose</p>
                          <p className="text-2xl font-semibold">
                            {lastPrediction.input.current_glucose_mgdl} mg/dL
                          </p>
                        </div>

                        <div className="pt-4 border-t">
                          <p className="text-sm font-medium mb-2">Explanation</p>
                          <p className="text-sm text-muted-foreground">
                            {lastPrediction.result.explanation}
                          </p>
                        </div>

                        <div className="pt-4 border-t">
                          <p className="text-xs text-muted-foreground">
                            {format(new Date(lastPrediction.timestamp), 'PPpp')}
                          </p>
                        </div>
                      </div>
                    ) : (
                      <div className="text-center py-8 text-muted-foreground">
                        <p>No predictions yet</p>
                        <p className="text-sm mt-2">Submit a prediction to see results here</p>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>

              {predictionHistory.length > 0 && (
                <>
                  <div className="px-4 lg:px-6">
                    <ChartAreaInteractive 
                      predictionHistory={predictionHistory}
                    />
                  </div>
                  <DataTable 
                    predictionHistory={predictionHistory}
                    getRiskBadgeVariant={getRiskBadgeVariant}
                    getRiskBadgeColor={getRiskBadgeColor}
                  />
                </>
              )}
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
