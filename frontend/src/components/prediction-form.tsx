import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import { useMutation } from '@tanstack/react-query'
import { AlertCircle } from 'lucide-react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
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
import { predictGlucosePredictPostMutation } from '@/client/@tanstack/react-query.gen'
import { savePredictionEntry } from '@/lib/storage'

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

interface PredictionFormProps {
  onSuccess?: () => void
}

export function PredictionForm({ onSuccess }: PredictionFormProps) {
  const form = useForm<PredictionFormValues>({
    resolver: zodResolver(predictionFormSchema),
    defaultValues: {
      current_glucose_mgdl: 139.2,
      raw_meal_exercise_text: '',
    },
  })

  const mutation = useMutation({
    ...predictGlucosePredictPostMutation(),
    onSuccess: (data) => {
      const input = form.getValues()
      // Save to localStorage
      savePredictionEntry(input, data)
      // Reset form
      form.reset({
        current_glucose_mgdl: undefined,
        raw_meal_exercise_text: '',
      })
      // Show toast notification
      const riskLabel = data.risk_label.toLowerCase()
      const isCritical = riskLabel === 'hypo' || riskLabel === 'hyper'
      const toastType = isCritical ? 'warning' : 'success'
      
      toast[toastType](
        `Prediction Complete: ${data.predicted_glucose_mgdl.toFixed(1)} mg/dL`,
        {
          description: `Risk Level: ${data.risk_label}${isCritical ? ' - Please take action' : ''}`,
          duration: isCritical ? 6000 : 4000,
        }
      )
      // Call success callback if provided
      if (onSuccess) {
        onSuccess()
      }
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

  return (
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
                  <FormLabel>This Morning's Glucose Reading (mg/dL)</FormLabel>
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
  )
}

