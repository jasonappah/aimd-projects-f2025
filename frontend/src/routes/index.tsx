import { createFileRoute } from '@tanstack/react-router'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, ReferenceLine } from 'recharts'
import { format } from 'date-fns'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from '@/components/ui/form'
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
  ChartLegend,
  ChartLegendContent,
} from '@/components/ui/chart'
import { predictGlucosePredictPostMutation } from '@/client/@tanstack/react-query.gen'
import { savePredictionEntry, getPredictionEntries } from '@/lib/storage'
import type { PredictionEntry } from '@/lib/storage'
import { AlertCircle } from 'lucide-react'

export const Route = createFileRoute('/')({
  component: App,
})

// Form validation schema
const predictionFormSchema = z.object({
  current_glucose_mgdl: z
    .number({
      required_error: 'Current glucose is required',
      invalid_type_error: 'Must be a number',
    })
    .min(0, 'Glucose must be at least 0')
    .max(500, 'Glucose must be at most 500'),
  raw_meal_exercise_text: z
    .string()
    .min(1, 'Meal/exercise description is required')
    .max(1000, 'Description is too long'),
})

type PredictionFormValues = z.infer<typeof predictionFormSchema>

function App() {
  const [predictionHistory, setPredictionHistory] = useState<PredictionEntry[]>([])
  const [lastPrediction, setLastPrediction] = useState<PredictionEntry | null>(null)

  // Load history from localStorage on mount
  useEffect(() => {
    const entries = getPredictionEntries()
    setPredictionHistory(entries)
    if (entries.length > 0) {
      setLastPrediction(entries[0])
    }
  }, [])

  const form = useForm<PredictionFormValues>({
    resolver: zodResolver(predictionFormSchema),
    defaultValues: {
      current_glucose_mgdl: undefined,
      raw_meal_exercise_text: '',
    },
  })

  const mutation = useMutation({
    ...predictGlucosePredictPostMutation(),
    onSuccess: (data) => {
      const input = form.getValues()
      // Save to localStorage
      savePredictionEntry(input, data)
      // Update state
      const entries = getPredictionEntries()
      setPredictionHistory(entries)
      if (entries.length > 0) {
        setLastPrediction(entries[0])
      }
      // Reset form
      form.reset({
        current_glucose_mgdl: undefined,
        raw_meal_exercise_text: '',
      })
    },
  })

  const onSubmit = (values: PredictionFormValues) => {
    mutation.mutate({
      body: {
        current_glucose_mgdl: values.current_glucose_mgdl,
        raw_meal_exercise_text: values.raw_meal_exercise_text,
      },
    })
  }

  // Prepare chart data (last 20 entries, reversed for chronological order)
  const chartData = predictionHistory
    .slice(0, 20)
    .reverse()
    .map((entry) => ({
      timestamp: format(new Date(entry.timestamp), 'MM/dd HH:mm'),
      current: entry.input.current_glucose_mgdl,
      predicted: entry.result.predicted_glucose_mgdl,
      risk: entry.result.risk_label,
    }))

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

  const chartConfig = {
    current: {
      label: 'Current Glucose',
      color: 'hsl(var(--chart-1))',
    },
    predicted: {
      label: 'Predicted Glucose',
      color: 'hsl(var(--chart-2))',
    },
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 py-8 px-4">
      <div className="max-w-6xl mx-auto space-y-8">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-4xl font-bold text-white mb-2">Blood Sugar Prediction</h1>
          <p className="text-gray-400">
            Enter your current glucose level and meal/exercise details to predict your blood sugar
            levels
          </p>
        </div>

        <div className="grid gap-8 md:grid-cols-2">
          {/* Form Section */}
          <Card>
            <CardHeader>
              <CardTitle>New Prediction</CardTitle>
              <CardDescription>
                Enter your current glucose reading and describe your meal or exercise
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Form {...form}>
                <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-6">
                  <FormField
                    control={form.control}
                    name="current_glucose_mgdl"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Current Glucose (mg/dL)</FormLabel>
                        <FormControl>
                          <Input
                            type="number"
                            placeholder="e.g., 120"
                            {...field}
                            onChange={(e) => field.onChange(parseFloat(e.target.value) || 0)}
                          />
                        </FormControl>
                        <FormDescription>
                          Your current blood glucose reading in mg/dL
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  <FormField
                    control={form.control}
                    name="raw_meal_exercise_text"
                    render={({ field }) => (
                      <FormItem>
                        <FormLabel>Meal or Exercise Description</FormLabel>
                        <FormControl>
                          <Textarea
                            placeholder="e.g., Ate a large pizza with 2 slices, going for a 30-minute walk"
                            className="min-h-[100px]"
                            {...field}
                          />
                        </FormControl>
                        <FormDescription>
                          Describe what you ate or any exercise you plan to do
                        </FormDescription>
                        <FormMessage />
                      </FormItem>
                    )}
                  />

                  {mutation.isError && (
                    <Alert variant="destructive">
                      <AlertCircle className="h-4 w-4" />
                      <AlertTitle>Error</AlertTitle>
                      <AlertDescription>
                        {mutation.error instanceof Error
                          ? mutation.error.message
                          : 'Failed to get prediction. Please try again.'}
                      </AlertDescription>
                    </Alert>
                  )}

                  <Button type="submit" disabled={mutation.isPending} className="w-full">
                    {mutation.isPending ? 'Predicting...' : 'Get Prediction'}
                  </Button>
                </form>
              </Form>
            </CardContent>
          </Card>

          {/* Results Section */}
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

        {/* History Chart Section */}
        {predictionHistory.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Prediction History</CardTitle>
              <CardDescription>
                Track your current and predicted glucose levels over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ChartContainer config={chartConfig} className="h-[400px] w-full">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis
                    dataKey="timestamp"
                    tick={{ fill: 'hsl(var(--muted-foreground))' }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis
                    tick={{ fill: 'hsl(var(--muted-foreground))' }}
                    label={{
                      value: 'Glucose (mg/dL)',
                      angle: -90,
                      position: 'insideLeft',
                      style: { fill: 'hsl(var(--muted-foreground))' },
                    }}
                  />
                  <ChartTooltip content={<ChartTooltipContent />} />
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
        )}
      </div>
    </div>
  )
}
