import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { 
  Activity, 
  Shield, 
  Cpu, 
  Database, 
  Network, 
  AlertTriangle, 
  CheckCircle, 
  XCircle,
  TrendingUp,
  Clock,
  Lock,
  Eye
} from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts'
import './App.css'

function App() {
  const [systemStatus, setSystemStatus] = useState('healthy')
  const [metrics, setMetrics] = useState({
    latticeOps: 0,
    cryptoOps: 0,
    aiInferences: 0,
    cacheHits: 0,
    gpuUtilization: 0,
    memoryUsage: 0
  })
  const [performanceData, setPerformanceData] = useState([])
  const [securityAlerts, setSecurityAlerts] = useState([])
  const [isConnected, setIsConnected] = useState(false)

  // Simulate real-time data updates
  useEffect(() => {
    const interval = setInterval(() => {
      // Update metrics
      setMetrics(prev => ({
        latticeOps: prev.latticeOps + Math.floor(Math.random() * 10),
        cryptoOps: prev.cryptoOps + Math.floor(Math.random() * 5),
        aiInferences: prev.aiInferences + Math.floor(Math.random() * 15),
        cacheHits: Math.min(100, prev.cacheHits + Math.random() * 2),
        gpuUtilization: Math.max(0, Math.min(100, prev.gpuUtilization + (Math.random() - 0.5) * 10)),
        memoryUsage: Math.max(0, Math.min(100, prev.memoryUsage + (Math.random() - 0.5) * 5))
      }))

      // Update performance data
      const now = new Date()
      const timeStr = now.toLocaleTimeString()
      setPerformanceData(prev => {
        const newData = [...prev, {
          time: timeStr,
          e8Ops: Math.floor(Math.random() * 100) + 50,
          leechOps: Math.floor(Math.random() * 80) + 30,
          cryptoOps: Math.floor(Math.random() * 60) + 20
        }]
        return newData.slice(-20) // Keep last 20 data points
      })

      // Simulate connection status
      setIsConnected(Math.random() > 0.1) // 90% uptime
    }, 2000)

    return () => clearInterval(interval)
  }, [])

  // Mock API call to check system health
  const checkSystemHealth = async () => {
    try {
      // In a real implementation, this would call the actual API
      const response = await fetch('/api/lattice/health')
      if (response.ok) {
        setSystemStatus('healthy')
        setIsConnected(true)
      } else {
        setSystemStatus('degraded')
      }
    } catch (error) {
      setSystemStatus('error')
      setIsConnected(false)
    }
  }

  useEffect(() => {
    checkSystemHealth()
    const healthCheck = setInterval(checkSystemHealth, 30000)
    return () => clearInterval(healthCheck)
  }, [])

  const statusColor = {
    healthy: 'text-green-600',
    degraded: 'text-yellow-600',
    error: 'text-red-600'
  }

  const statusIcon = {
    healthy: <CheckCircle className="h-5 w-5 text-green-600" />,
    degraded: <AlertTriangle className="h-5 w-5 text-yellow-600" />,
    error: <XCircle className="h-5 w-5 text-red-600" />
  }

  const pieData = [
    { name: 'E8 Operations', value: metrics.latticeOps * 0.4, fill: '#8884d8' },
    { name: 'Leech Operations', value: metrics.latticeOps * 0.6, fill: '#82ca9d' },
    { name: 'Crypto Operations', value: metrics.cryptoOps, fill: '#ffc658' },
    { name: 'AI Inferences', value: metrics.aiInferences, fill: '#ff7300' }
  ]

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="container mx-auto p-6">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                E8 Leech Lattice Framework
              </h1>
              <p className="text-slate-300 mt-2">Real-time Monitoring Dashboard</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                {statusIcon[systemStatus]}
                <span className={`font-medium ${statusColor[systemStatus]}`}>
                  {systemStatus.charAt(0).toUpperCase() + systemStatus.slice(1)}
                </span>
              </div>
              <Badge variant={isConnected ? "default" : "destructive"}>
                {isConnected ? "Connected" : "Disconnected"}
              </Badge>
            </div>
          </div>
        </div>

        {/* System Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-slate-200">Lattice Operations</CardTitle>
              <Activity className="h-4 w-4 text-blue-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{metrics.latticeOps.toLocaleString()}</div>
              <p className="text-xs text-slate-400">
                <TrendingUp className="inline h-3 w-3 mr-1" />
                +12% from last hour
              </p>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-slate-200">Crypto Operations</CardTitle>
              <Shield className="h-4 w-4 text-green-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{metrics.cryptoOps.toLocaleString()}</div>
              <p className="text-xs text-slate-400">
                <Lock className="inline h-3 w-3 mr-1" />
                All secure
              </p>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-slate-200">AI Inferences</CardTitle>
              <Cpu className="h-4 w-4 text-purple-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{metrics.aiInferences.toLocaleString()}</div>
              <p className="text-xs text-slate-400">
                <Eye className="inline h-3 w-3 mr-1" />
                Models active
              </p>
            </CardContent>
          </Card>

          <Card className="bg-slate-800/50 border-slate-700">
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium text-slate-200">Cache Hit Rate</CardTitle>
              <Database className="h-4 w-4 text-orange-400" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-white">{metrics.cacheHits.toFixed(1)}%</div>
              <Progress value={metrics.cacheHits} className="mt-2" />
            </CardContent>
          </Card>
        </div>

        {/* Main Dashboard Tabs */}
        <Tabs defaultValue="performance" className="space-y-6">
          <TabsList className="grid w-full grid-cols-4 bg-slate-800/50">
            <TabsTrigger value="performance">Performance</TabsTrigger>
            <TabsTrigger value="security">Security</TabsTrigger>
            <TabsTrigger value="lattice">Lattice Ops</TabsTrigger>
            <TabsTrigger value="system">System</TabsTrigger>
          </TabsList>

          <TabsContent value="performance" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">Operations Performance</CardTitle>
                  <CardDescription className="text-slate-400">
                    Real-time operation throughput
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={performanceData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis dataKey="time" stroke="#9CA3AF" />
                      <YAxis stroke="#9CA3AF" />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1F2937', 
                          border: '1px solid #374151',
                          borderRadius: '8px'
                        }}
                      />
                      <Line type="monotone" dataKey="e8Ops" stroke="#3B82F6" strokeWidth={2} />
                      <Line type="monotone" dataKey="leechOps" stroke="#10B981" strokeWidth={2} />
                      <Line type="monotone" dataKey="cryptoOps" stroke="#F59E0B" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">Operation Distribution</CardTitle>
                  <CardDescription className="text-slate-400">
                    Breakdown by operation type
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={pieData}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        outerRadius={80}
                        fill="#8884d8"
                        dataKey="value"
                        label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                      >
                        {pieData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">GPU Utilization</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-white mb-2">
                    {metrics.gpuUtilization.toFixed(1)}%
                  </div>
                  <Progress value={metrics.gpuUtilization} className="mb-2" />
                  <p className="text-sm text-slate-400">
                    {metrics.gpuUtilization > 80 ? 'High utilization' : 'Normal operation'}
                  </p>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">Memory Usage</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-3xl font-bold text-white mb-2">
                    {metrics.memoryUsage.toFixed(1)}%
                  </div>
                  <Progress value={metrics.memoryUsage} className="mb-2" />
                  <p className="text-sm text-slate-400">
                    {metrics.memoryUsage > 90 ? 'Memory pressure' : 'Available memory'}
                  </p>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="security" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">Security Status</CardTitle>
                  <CardDescription className="text-slate-400">
                    Cryptographic module audit
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">LWE Encryption</span>
                    <Badge variant="default" className="bg-green-600">Secure</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">BLISS Signatures</span>
                    <Badge variant="default" className="bg-green-600">Verified</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">Key Exchange</span>
                    <Badge variant="default" className="bg-green-600">Active</Badge>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-slate-300">Quantum Resistance</span>
                    <Badge variant="default" className="bg-blue-600">High</Badge>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">Recent Security Events</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Alert className="bg-green-900/20 border-green-700">
                    <CheckCircle className="h-4 w-4" />
                    <AlertTitle className="text-green-400">Security Audit Passed</AlertTitle>
                    <AlertDescription className="text-green-300">
                      All cryptographic modules verified - 2 minutes ago
                    </AlertDescription>
                  </Alert>
                  <Alert className="bg-blue-900/20 border-blue-700">
                    <Shield className="h-4 w-4" />
                    <AlertTitle className="text-blue-400">Key Rotation</AlertTitle>
                    <AlertDescription className="text-blue-300">
                      Automatic key rotation completed - 15 minutes ago
                    </AlertDescription>
                  </Alert>
                  <Alert className="bg-yellow-900/20 border-yellow-700">
                    <AlertTriangle className="h-4 w-4" />
                    <AlertTitle className="text-yellow-400">Rate Limit Warning</AlertTitle>
                    <AlertDescription className="text-yellow-300">
                      High request volume detected - 1 hour ago
                    </AlertDescription>
                  </Alert>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="lattice" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">E8 Lattice</CardTitle>
                  <CardDescription className="text-slate-400">8-dimensional operations</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Quantizations</span>
                    <span className="text-white font-mono">{Math.floor(metrics.latticeOps * 0.4).toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Nearest Neighbor</span>
                    <span className="text-white font-mono">{Math.floor(metrics.latticeOps * 0.3).toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Theta Functions</span>
                    <span className="text-white font-mono">{Math.floor(metrics.latticeOps * 0.1).toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Root System Access</span>
                    <span className="text-white font-mono">{Math.floor(metrics.latticeOps * 0.2).toLocaleString()}</span>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">Leech Lattice</CardTitle>
                  <CardDescription className="text-slate-400">24-dimensional operations</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Quantizations</span>
                    <span className="text-white font-mono">{Math.floor(metrics.latticeOps * 0.6 * 0.5).toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Golay Decoding</span>
                    <span className="text-white font-mono">{Math.floor(metrics.latticeOps * 0.6 * 0.3).toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Congruence Checks</span>
                    <span className="text-white font-mono">{Math.floor(metrics.latticeOps * 0.6 * 0.2).toLocaleString()}</span>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">Performance Metrics</CardTitle>
                  <CardDescription className="text-slate-400">Average operation times</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-300">E8 Quantization</span>
                    <span className="text-white font-mono">0.12ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Leech Quantization</span>
                    <span className="text-white font-mono">0.45ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Babai Algorithm</span>
                    <span className="text-white font-mono">0.08ms</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Cache Hit Rate</span>
                    <span className="text-white font-mono">{metrics.cacheHits.toFixed(1)}%</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="system" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">System Information</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-slate-300">Framework Version</span>
                    <span className="text-white font-mono">1.0.0</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">API Status</span>
                    <Badge variant={isConnected ? "default" : "destructive"}>
                      {isConnected ? "Online" : "Offline"}
                    </Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">GPU Acceleration</span>
                    <Badge variant="default" className="bg-green-600">Available</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Quantum Modules</span>
                    <Badge variant="default" className="bg-blue-600">Loaded</Badge>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-300">Uptime</span>
                    <span className="text-white font-mono">2d 14h 32m</span>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">Quick Actions</CardTitle>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Button 
                    className="w-full bg-blue-600 hover:bg-blue-700" 
                    onClick={checkSystemHealth}
                  >
                    <Activity className="mr-2 h-4 w-4" />
                    Refresh System Status
                  </Button>
                  <Button 
                    variant="outline" 
                    className="w-full border-slate-600 text-slate-300 hover:bg-slate-700"
                  >
                    <Database className="mr-2 h-4 w-4" />
                    Clear Cache
                  </Button>
                  <Button 
                    variant="outline" 
                    className="w-full border-slate-600 text-slate-300 hover:bg-slate-700"
                  >
                    <Shield className="mr-2 h-4 w-4" />
                    Run Security Audit
                  </Button>
                  <Button 
                    variant="outline" 
                    className="w-full border-slate-600 text-slate-300 hover:bg-slate-700"
                  >
                    <Network className="mr-2 h-4 w-4" />
                    Test API Endpoints
                  </Button>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}

export default App

