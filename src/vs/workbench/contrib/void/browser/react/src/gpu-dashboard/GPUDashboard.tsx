/*--------------------------------------------------------------------------------------
 *  Copyright 2025 VS Aware Development Team. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle, Button } from '../util/components';
import { 
	Cpu, 
	Thermometer, 
	Zap, 
	Monitor, 
	AlertTriangle, 
	AlertCircle,
	Activity,
	X,
	Settings,
	Pause,
	Play
} from 'lucide-react';

interface GPUInfo {
	id: number;
	name: string;
	driverVersion: string;
	memoryTotal: number;
	computeCapability: string;
	temperature: number;
	powerDraw: number;
	powerLimit: number;
}

interface GPUUtilization {
	gpuId: number;
	gpuUtilization: number;
	memoryUtilization: number;
	memoryUsed: number;
	memoryFree: number;
	temperature: number;
	powerDraw: number;
	processes: GPUProcess[];
}

interface GPUProcess {
	pid: number;
	name: string;
	memoryUsage: number;
	gpuUtilization: number;
	type: 'C' | 'G' | 'C+G';
}

interface ResourceAlert {
	type: 'memory' | 'temperature' | 'power';
	severity: 'warning' | 'critical';
	message: string;
	gpuId?: number;
	threshold: number;
	current: number;
	timestamp: Date;
}

interface GPUDashboardProps {
	gpus: GPUInfo[];
	utilization: Map<number, GPUUtilization>;
	alerts: ResourceAlert[];
	isMonitoring: boolean;
	onStartMonitoring: () => void;
	onStopMonitoring: () => void;
	onKillProcess: (pid: number) => Promise<boolean>;
	onUpdateThresholds: (thresholds: any) => void;
}

const GPUDashboard: React.FC<GPUDashboardProps> = ({
	gpus,
	utilization,
	alerts,
	isMonitoring,
	onStartMonitoring,
	onStopMonitoring,
	onKillProcess,
	onUpdateThresholds
}) => {
	const [selectedGPU, setSelectedGPU] = useState<number>(0);
	const [showSettings, setShowSettings] = useState(false);
	const [showAlerts, setShowAlerts] = useState(true);
	const chartRef = useRef<HTMLCanvasElement>(null);

	const selectedUtilization = utilization.get(selectedGPU);

	const formatMemory = (mb: number) => {
		if (mb >= 1024) {
			return `${(mb / 1024).toFixed(1)} GB`;
		}
		return `${mb} MB`;
	};

	const getUtilizationColor = (value: number, type: 'memory' | 'gpu' | 'temperature') => {
		if (type === 'temperature') {
			if (value >= 90) return 'text-red-600';
			if (value >= 80) return 'text-yellow-600';
			return 'text-green-600';
		} else {
			if (value >= 90) return 'text-red-600';
			if (value >= 70) return 'text-yellow-600';
			return 'text-green-600';
		}
	};

	const getProgressBarColor = (value: number, type: 'memory' | 'gpu' | 'temperature') => {
		if (type === 'temperature') {
			if (value >= 90) return 'bg-red-500';
			if (value >= 80) return 'bg-yellow-500';
			return 'bg-green-500';
		} else {
			if (value >= 90) return 'bg-red-500';
			if (value >= 70) return 'bg-yellow-500';
			return 'bg-blue-500';
		}
	};

	const ProgressBar: React.FC<{ value: number; max: number; type: 'memory' | 'gpu' | 'temperature' }> = ({ value, max, type }) => {
		const percentage = (value / max) * 100;
		return (
			<div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
				<div
					className={`h-2 rounded-full transition-all duration-300 ${getProgressBarColor(percentage, type)}`}
					style={{ width: `${Math.min(percentage, 100)}%` }}
				/>
			</div>
		);
	};

	const MetricCard: React.FC<{ 
		title: string; 
		value: string | number; 
		unit?: string; 
		icon: React.ReactNode; 
		progress?: { value: number; max: number; type: 'memory' | 'gpu' | 'temperature' }
	}> = ({ title, value, unit, icon, progress }) => (
		<div className="bg-white dark:bg-gray-800 p-4 rounded-lg border dark:border-gray-700">
			<div className="flex items-center justify-between mb-2">
				<div className="flex items-center space-x-2">
					{icon}
					<span className="text-sm font-medium text-gray-600 dark:text-gray-400">{title}</span>
				</div>
			</div>
			<div className="flex items-baseline space-x-1">
				<span className={`text-2xl font-bold ${progress ? getUtilizationColor(progress.value, progress.type) : ''}`}>
					{value}
				</span>
				{unit && <span className="text-sm text-gray-500 dark:text-gray-400">{unit}</span>}
			</div>
			{progress && (
				<div className="mt-2">
					<ProgressBar value={progress.value} max={progress.max} type={progress.type} />
				</div>
			)}
		</div>
	);

	return (
		<div className="gpu-dashboard p-6 space-y-6 max-w-7xl mx-auto">
			{/* Header */}
			<div className="flex items-center justify-between">
				<div>
					<h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
						<Monitor className="inline mr-2 h-8 w-8 text-green-600" />
						GPU Resource Dashboard
					</h1>
					<p className="text-gray-600 dark:text-gray-300">
						Real-time monitoring of GPU and TPU resources
					</p>
				</div>
				
				<div className="flex items-center space-x-3">
					<Button
						onClick={showSettings ? () => setShowSettings(false) : () => setShowSettings(true)}
						variant="outline"
						size="sm"
					>
						<Settings className="h-4 w-4 mr-2" />
						Settings
					</Button>
					
					<Button
						onClick={isMonitoring ? onStopMonitoring : onStartMonitoring}
						variant={isMonitoring ? "destructive" : "default"}
						size="sm"
					>
						{isMonitoring ? (
							<>
								<Pause className="h-4 w-4 mr-2" />
								Stop Monitoring
							</>
						) : (
							<>
								<Play className="h-4 w-4 mr-2" />
								Start Monitoring
							</>
						)}
					</Button>
				</div>
			</div>

			{/* Alerts */}
			{showAlerts && alerts.length > 0 && (
				<div className="space-y-2">
					{alerts.slice(0, 3).map((alert, index) => (
						<div
							key={index}
							className={`p-3 rounded-lg border flex items-center justify-between ${
								alert.severity === 'critical' 
									? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800' 
									: 'bg-yellow-50 dark:bg-yellow-900/20 border-yellow-200 dark:border-yellow-800'
							}`}
						>
							<div className="flex items-center space-x-2">
								{alert.severity === 'critical' ? (
									<AlertCircle className="h-5 w-5 text-red-600" />
								) : (
									<AlertTriangle className="h-5 w-5 text-yellow-600" />
								)}
								<span className="font-medium">{alert.message}</span>
								<span className="text-sm text-gray-500">
									({alert.current.toFixed(1)}% / {alert.threshold}%)
								</span>
							</div>
							<Button
								onClick={() => setShowAlerts(false)}
								variant="ghost"
								size="sm"
							>
								<X className="h-4 w-4" />
							</Button>
						</div>
					))}
				</div>
			)}

			{/* GPU Selection */}
			{gpus.length > 1 && (
				<div className="flex space-x-2 overflow-x-auto">
					{gpus.map((gpu) => (
						<button
							key={gpu.id}
							onClick={() => setSelectedGPU(gpu.id)}
							className={`px-4 py-2 rounded-lg whitespace-nowrap text-sm font-medium transition-colors ${
								selectedGPU === gpu.id
									? 'bg-blue-600 text-white'
									: 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
							}`}
						>
							GPU {gpu.id}: {gpu.name}
						</button>
					))}
				</div>
			)}

			{/* Main Metrics */}
			{selectedUtilization && (
				<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
					<MetricCard
						title="GPU Utilization"
						value={selectedUtilization.gpuUtilization}
						unit="%"
						icon={<Cpu className="h-5 w-5 text-blue-600" />}
						progress={{ value: selectedUtilization.gpuUtilization, max: 100, type: 'gpu' }}
					/>
					
					<MetricCard
						title="Memory Usage"
						value={`${selectedUtilization.memoryUtilization}%`}
						unit={`${formatMemory(selectedUtilization.memoryUsed)} / ${formatMemory(selectedUtilization.memoryUsed + selectedUtilization.memoryFree)}`}
						icon={<Activity className="h-5 w-5 text-purple-600" />}
						progress={{ value: selectedUtilization.memoryUtilization, max: 100, type: 'memory' }}
					/>
					
					<MetricCard
						title="Temperature"
						value={selectedUtilization.temperature}
						unit="°C"
						icon={<Thermometer className="h-5 w-5 text-red-600" />}
						progress={{ value: selectedUtilization.temperature, max: 100, type: 'temperature' }}
					/>
					
					<MetricCard
						title="Power Draw"
						value={selectedUtilization.powerDraw}
						unit="W"
						icon={<Zap className="h-5 w-5 text-yellow-600" />}
						progress={{ 
							value: selectedUtilization.powerDraw, 
							max: gpus.find(g => g.id === selectedGPU)?.powerLimit || 300, 
							type: 'gpu' 
						}}
					/>
				</div>
			)}

			{/* Detailed View */}
			<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
				{/* GPU Processes */}
				<Card>
					<CardHeader>
						<CardTitle className="flex items-center">
							<Activity className="mr-2 h-5 w-5" />
							Running Processes
						</CardTitle>
					</CardHeader>
					<CardContent>
						{selectedUtilization?.processes.length ? (
							<div className="space-y-3">
								{selectedUtilization.processes.map((process) => (
									<div
										key={process.pid}
										className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800 rounded-lg"
									>
										<div className="flex-1">
											<div className="flex items-center justify-between">
												<span className="font-medium">{process.name}</span>
												<span className="text-sm text-gray-500">PID: {process.pid}</span>
											</div>
											<div className="flex items-center space-x-4 mt-1 text-sm text-gray-600 dark:text-gray-400">
												<span>Memory: {formatMemory(process.memoryUsage)}</span>
												<span>GPU: {process.gpuUtilization}%</span>
												<span className="px-2 py-1 bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 rounded text-xs">
													{process.type}
												</span>
											</div>
										</div>
										<Button
											onClick={() => onKillProcess(process.pid)}
											variant="destructive"
											size="sm"
										>
											<X className="h-4 w-4" />
										</Button>
									</div>
								))}
							</div>
						) : (
							<div className="text-center py-8 text-gray-500 dark:text-gray-400">
								No GPU processes running
							</div>
						)}
					</CardContent>
				</Card>

				{/* Historical Chart */}
				<Card>
					<CardHeader>
						<CardTitle className="flex items-center">
							<Activity className="mr-2 h-5 w-5" />
							Historical Usage
						</CardTitle>
					</CardHeader>
					<CardContent>
						<div className="h-64 flex items-center justify-center text-gray-500 dark:text-gray-400">
							<canvas ref={chartRef} className="w-full h-full" />
							<div className="text-center">
								<Activity className="h-12 w-12 mx-auto mb-2 text-gray-400" />
								<p>Historical usage chart</p>
								<p className="text-sm">Chart implementation needed</p>
							</div>
						</div>
					</CardContent>
				</Card>
			</div>

			{/* GPU Details */}
			{gpus.find(g => g.id === selectedGPU) && (
				<Card>
					<CardHeader>
						<CardTitle>GPU {selectedGPU} Details</CardTitle>
					</CardHeader>
					<CardContent>
						<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
							{(() => {
								const gpu = gpus.find(g => g.id === selectedGPU)!;
								return (
									<>
										<div>
											<label className="text-sm font-medium text-gray-600 dark:text-gray-400">Name</label>
											<p className="mt-1 font-medium">{gpu.name}</p>
										</div>
										<div>
											<label className="text-sm font-medium text-gray-600 dark:text-gray-400">Driver Version</label>
											<p className="mt-1 font-medium">{gpu.driverVersion}</p>
										</div>
										<div>
											<label className="text-sm font-medium text-gray-600 dark:text-gray-400">Total Memory</label>
											<p className="mt-1 font-medium">{formatMemory(gpu.memoryTotal)}</p>
										</div>
										<div>
											<label className="text-sm font-medium text-gray-600 dark:text-gray-400">Compute Capability</label>
											<p className="mt-1 font-medium">{gpu.computeCapability}</p>
										</div>
									</>
								);
							})()}
						</div>
					</CardContent>
				</Card>
			)}

			{/* Settings Panel */}
			{showSettings && (
				<Card>
					<CardHeader>
						<CardTitle className="flex items-center justify-between">
							<span>Dashboard Settings</span>
							<Button onClick={() => setShowSettings(false)} variant="ghost" size="sm">
								<X className="h-4 w-4" />
							</Button>
						</CardTitle>
					</CardHeader>
					<CardContent>
						<div className="space-y-4">
							<div>
								<label className="block text-sm font-medium mb-2">Alert Thresholds</label>
								<div className="grid grid-cols-1 md:grid-cols-3 gap-4">
									<div>
										<label className="text-xs text-gray-600 dark:text-gray-400">Memory Warning (%)</label>
										<input type="number" defaultValue={80} className="w-full mt-1 p-2 border rounded dark:bg-gray-800 dark:border-gray-600" />
									</div>
									<div>
										<label className="text-xs text-gray-600 dark:text-gray-400">Memory Critical (%)</label>
										<input type="number" defaultValue={95} className="w-full mt-1 p-2 border rounded dark:bg-gray-800 dark:border-gray-600" />
									</div>
									<div>
										<label className="text-xs text-gray-600 dark:text-gray-400">Temperature Warning (°C)</label>
										<input type="number" defaultValue={80} className="w-full mt-1 p-2 border rounded dark:bg-gray-800 dark:border-gray-600" />
									</div>
								</div>
							</div>
							
							<div className="flex items-center space-x-2">
								<input type="checkbox" id="auto-refresh" defaultChecked />
								<label htmlFor="auto-refresh" className="text-sm">Auto-refresh every 2 seconds</label>
							</div>
							
							<div className="flex items-center space-x-2">
								<input type="checkbox" id="show-notifications" defaultChecked />
								<label htmlFor="show-notifications" className="text-sm">Show desktop notifications for alerts</label>
							</div>
						</div>
					</CardContent>
				</Card>
			)}
		</div>
	);
};

export default GPUDashboard;