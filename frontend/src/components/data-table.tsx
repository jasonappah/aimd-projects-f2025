import * as React from "react"
import {
  ColumnDef,
  ColumnFiltersState,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  SortingState,
  useReactTable,
  VisibilityState,
} from "@tanstack/react-table"
import { format } from "date-fns"
import {
  IconChevronDown,
  IconChevronLeft,
  IconChevronRight,
  IconChevronsLeft,
  IconChevronsRight,
  IconLayoutColumns,
} from "@tabler/icons-react"

import { useIsMobile } from "@/hooks/use-mobile"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import {
  Drawer,
  DrawerClose,
  DrawerContent,
  DrawerDescription,
  DrawerHeader,
  DrawerTitle,
  DrawerTrigger,
} from "@/components/ui/drawer"
import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import type { PredictionEntry } from "@/lib/storage"

interface DataTableProps {
  predictionHistory: PredictionEntry[]
  getRiskBadgeVariant?: (riskLabel: string) => 'default' | 'destructive' | 'outline'
  getRiskBadgeColor?: (riskLabel: string) => string
}

function PredictionDetailViewer({ entry, getRiskBadgeVariant, getRiskBadgeColor }: { 
  entry: PredictionEntry
  getRiskBadgeVariant?: (riskLabel: string) => 'default' | 'destructive' | 'outline'
  getRiskBadgeColor?: (riskLabel: string) => string
}) {
  const isMobile = useIsMobile()

  const getRiskBadgeVariantDefault = (riskLabel: string): 'default' | 'destructive' | 'outline' => {
    if (getRiskBadgeVariant) return getRiskBadgeVariant(riskLabel)
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

  const getRiskBadgeColorDefault = (riskLabel: string): string => {
    if (getRiskBadgeColor) return getRiskBadgeColor(riskLabel)
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
    <Drawer direction={isMobile ? "bottom" : "right"}>
      <DrawerTrigger asChild>
        <Button variant="link" className="text-foreground w-fit px-0 text-left">
          View Details
        </Button>
      </DrawerTrigger>
      <DrawerContent>
        <DrawerHeader className="gap-1">
          <DrawerTitle>Prediction Details</DrawerTitle>
          <DrawerDescription>
            {format(new Date(entry.timestamp), 'PPpp')}
          </DrawerDescription>
        </DrawerHeader>
        <div className="flex flex-col gap-4 overflow-y-auto px-4 text-sm">
          <div className="grid gap-4">
            <div className="flex items-center justify-between">
              <div>
                <Label className="text-muted-foreground">This Morning's Glucose Reading</Label>
                <p className="text-2xl font-semibold">
                  {entry.input.current_glucose_mgdl} mg/dL
                </p>
              </div>
              <Badge
                variant={getRiskBadgeVariantDefault(entry.result.risk_label)}
                className={`${getRiskBadgeColorDefault(entry.result.risk_label)} text-white`}
              >
                {entry.result.risk_label}
              </Badge>
            </div>
            <div>
              <Label className="text-muted-foreground">Predicted Glucose</Label>
              <p className="text-2xl font-semibold">
                {entry.result.predicted_glucose_mgdl.toFixed(1)} mg/dL
              </p>
            </div>
            <div>
              <Label className="text-muted-foreground">Meal/Exercise Description</Label>
              <p className="text-sm">{entry.input.raw_meal_exercise_text}</p>
            </div>
            <div>
              <Label className="text-muted-foreground">Explanation</Label>
              <p className="text-sm">{entry.result.explanation}</p>
            </div>
          </div>
        </div>
        <div className="px-4 pb-4">
          <DrawerClose asChild>
            <Button variant="outline" className="w-full">Close</Button>
          </DrawerClose>
        </div>
      </DrawerContent>
    </Drawer>
  )
}

const columns = (
  getRiskBadgeVariant?: (riskLabel: string) => 'default' | 'destructive' | 'outline',
  getRiskBadgeColor?: (riskLabel: string) => string
): ColumnDef<PredictionEntry>[] => [
  {
    id: "select",
    header: ({ table }) => (
      <div className="flex items-center justify-center">
        <Checkbox
          checked={
            table.getIsAllPageRowsSelected() ||
            (table.getIsSomePageRowsSelected() && "indeterminate")
          }
          onCheckedChange={(value) => table.toggleAllPageRowsSelected(!!value)}
          aria-label="Select all"
        />
      </div>
    ),
    cell: ({ row }) => (
      <div className="flex items-center justify-center">
        <Checkbox
          checked={row.getIsSelected()}
          onCheckedChange={(value) => row.toggleSelected(!!value)}
          aria-label="Select row"
        />
      </div>
    ),
    enableSorting: false,
    enableHiding: false,
  },
  {
    accessorKey: "timestamp",
    header: "Date & Time",
    cell: ({ row }) => {
      return (
        <div>
          <div className="font-medium">
            {format(new Date(row.original.timestamp), 'PPp')}
          </div>
          <div className="text-sm text-muted-foreground">
            {format(new Date(row.original.timestamp), 'MMM d, yyyy')}
          </div>
        </div>
      )
    },
    enableHiding: false,
  },
  {
    accessorKey: "input.current_glucose_mgdl",
    header: () => <div className="text-right">Current (mg/dL)</div>,
    cell: ({ row }) => {
      return (
        <div className="text-right font-medium">
          {row.original.input.current_glucose_mgdl}
        </div>
      )
    },
  },
  {
    accessorKey: "result.predicted_glucose_mgdl",
    header: () => <div className="text-right">Predicted (mg/dL)</div>,
    cell: ({ row }) => {
      return (
        <div className="text-right font-medium">
          {row.original.result.predicted_glucose_mgdl.toFixed(1)}
        </div>
      )
    },
  },
  {
    accessorKey: "result.risk_label",
    header: "Risk Level",
    cell: ({ row }) => {
      const riskLabel = row.original.result.risk_label
      const getRiskBadgeVariantDefault = (riskLabel: string): 'default' | 'destructive' | 'outline' => {
        if (getRiskBadgeVariant) return getRiskBadgeVariant(riskLabel)
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
      const getRiskBadgeColorDefault = (riskLabel: string): string => {
        if (getRiskBadgeColor) return getRiskBadgeColor(riskLabel)
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
        <Badge
          variant={getRiskBadgeVariantDefault(riskLabel)}
          className={`${getRiskBadgeColorDefault(riskLabel)} text-white`}
        >
          {riskLabel}
        </Badge>
      )
    },
  },
  {
    accessorKey: "input.raw_meal_exercise_text",
    header: "Description",
    cell: ({ row }) => {
      const text = row.original.input.raw_meal_exercise_text
      return (
        <div className="max-w-[300px] truncate" title={text}>
          {text}
        </div>
      )
    },
  },
  {
    id: "actions",
    cell: ({ row }) => {
      return (
        <PredictionDetailViewer
          entry={row.original}
          getRiskBadgeVariant={getRiskBadgeVariant}
          getRiskBadgeColor={getRiskBadgeColor}
        />
      )
    },
  },
]

export function DataTable({
  predictionHistory,
  getRiskBadgeVariant,
  getRiskBadgeColor,
}: DataTableProps) {
  const [rowSelection, setRowSelection] = React.useState({})
  const [columnVisibility, setColumnVisibility] = React.useState<VisibilityState>({})
  const [columnFilters, setColumnFilters] = React.useState<ColumnFiltersState>([])
  const [sorting, setSorting] = React.useState<SortingState>([
    { id: "timestamp", desc: true }, // Most recent first
  ])
  const [pagination, setPagination] = React.useState({
    pageIndex: 0,
    pageSize: 10,
  })

  const table = useReactTable({
    data: predictionHistory,
    columns: columns(getRiskBadgeVariant, getRiskBadgeColor),
    state: {
      sorting,
      columnVisibility,
      rowSelection,
      columnFilters,
      pagination,
    },
    getRowId: (row, index) => row.timestamp + index,
    enableRowSelection: true,
    onRowSelectionChange: setRowSelection,
    onSortingChange: setSorting,
    onColumnFiltersChange: setColumnFilters,
    onColumnVisibilityChange: setColumnVisibility,
    onPaginationChange: setPagination,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getSortedRowModel: getSortedRowModel(),
  })

  if (predictionHistory.length === 0) {
    return (
      <Card className="mx-4 lg:mx-6">
        <CardHeader>
          <CardTitle>Prediction History</CardTitle>
          <CardDescription>
            View all your blood sugar predictions in a detailed table
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-muted-foreground">
            <p>No predictions yet</p>
            <p className="text-sm mt-2">Submit a prediction to see it in the history table</p>
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="mx-4 lg:mx-6">
      <CardHeader>
        <CardTitle>Prediction History</CardTitle>
        <CardDescription>
          View all your blood sugar predictions in a detailed table
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          <div className="flex items-center justify-end">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" size="sm">
                  <IconLayoutColumns />
                  <span className="hidden lg:inline">Customize Columns</span>
                  <span className="lg:hidden">Columns</span>
                  <IconChevronDown />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end" className="w-56">
                {table
                  .getAllColumns()
                  .filter(
                    (column) =>
                      typeof column.accessorFn !== "undefined" &&
                      column.getCanHide()
                  )
                  .map((column) => {
                    return (
                      <DropdownMenuCheckboxItem
                        key={column.id}
                        className="capitalize"
                        checked={column.getIsVisible()}
                        onCheckedChange={(value) =>
                          column.toggleVisibility(!!value)
                        }
                      >
                        {column.id}
                      </DropdownMenuCheckboxItem>
                    )
                  })}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
          <div className="overflow-hidden rounded-lg border">
            <Table>
              <TableHeader className="bg-muted">
                {table.getHeaderGroups().map((headerGroup) => (
                  <TableRow key={headerGroup.id}>
                    {headerGroup.headers.map((header) => {
                      return (
                        <TableHead key={header.id} colSpan={header.colSpan}>
                          {header.isPlaceholder
                            ? null
                            : flexRender(
                                header.column.columnDef.header,
                                header.getContext()
                              )}
                        </TableHead>
                      )
                    })}
                  </TableRow>
                ))}
              </TableHeader>
              <TableBody>
                {table.getRowModel().rows?.length ? (
                  table.getRowModel().rows.map((row) => (
                    <TableRow
                      key={row.id}
                      data-state={row.getIsSelected() && "selected"}
                    >
                      {row.getVisibleCells().map((cell) => (
                        <TableCell key={cell.id}>
                          {flexRender(cell.column.columnDef.cell, cell.getContext())}
                        </TableCell>
                      ))}
                    </TableRow>
                  ))
                ) : (
                  <TableRow>
                    <TableCell
                      colSpan={columns().length}
                      className="h-24 text-center"
                    >
                      No results.
                    </TableCell>
                  </TableRow>
                )}
              </TableBody>
            </Table>
          </div>
          <div className="flex items-center justify-between px-4">
            <div className="text-muted-foreground hidden flex-1 text-sm lg:flex">
              {table.getFilteredSelectedRowModel().rows.length} of{" "}
              {table.getFilteredRowModel().rows.length} row(s) selected.
            </div>
            <div className="flex w-full items-center gap-8 lg:w-fit">
              <div className="hidden items-center gap-2 lg:flex">
                <Label htmlFor="rows-per-page" className="text-sm font-medium">
                  Rows per page
                </Label>
                <Select
                  value={`${table.getState().pagination.pageSize}`}
                  onValueChange={(value) => {
                    table.setPageSize(Number(value))
                  }}
                >
                  <SelectTrigger size="sm" className="w-20" id="rows-per-page">
                    <SelectValue
                      placeholder={table.getState().pagination.pageSize}
                    />
                  </SelectTrigger>
                  <SelectContent side="top">
                    {[10, 20, 30, 40, 50].map((pageSize) => (
                      <SelectItem key={pageSize} value={`${pageSize}`}>
                        {pageSize}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex w-fit items-center justify-center text-sm font-medium">
                Page {table.getState().pagination.pageIndex + 1} of{" "}
                {table.getPageCount()}
              </div>
              <div className="ml-auto flex items-center gap-2 lg:ml-0">
                <Button
                  variant="outline"
                  className="hidden h-8 w-8 p-0 lg:flex"
                  onClick={() => table.setPageIndex(0)}
                  disabled={!table.getCanPreviousPage()}
                >
                  <span className="sr-only">Go to first page</span>
                  <IconChevronsLeft />
                </Button>
                <Button
                  variant="outline"
                  className="size-8"
                  size="icon"
                  onClick={() => table.previousPage()}
                  disabled={!table.getCanPreviousPage()}
                >
                  <span className="sr-only">Go to previous page</span>
                  <IconChevronLeft />
                </Button>
                <Button
                  variant="outline"
                  className="size-8"
                  size="icon"
                  onClick={() => table.nextPage()}
                  disabled={!table.getCanNextPage()}
                >
                  <span className="sr-only">Go to next page</span>
                  <IconChevronRight />
                </Button>
                <Button
                  variant="outline"
                  className="hidden size-8 lg:flex"
                  size="icon"
                  onClick={() => table.setPageIndex(table.getPageCount() - 1)}
                  disabled={!table.getCanNextPage()}
                >
                  <span className="sr-only">Go to last page</span>
                  <IconChevronsRight />
                </Button>
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
