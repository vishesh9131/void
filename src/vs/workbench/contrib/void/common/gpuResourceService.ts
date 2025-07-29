/*--------------------------------------------------------------------------------------
 *  Copyright 2025 VS Aware Development Team. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

import { Event, Emitter } from '../../../../base/common/event.js';
import { Disposable, toDisposable } from '../../../../base/common/lifecycle.js';
import { createDecorator } from '../../../../platform/instantiation/common/instantiation.js';
import { registerSingleton, InstantiationType } from '../../../../platform/instantiation/common/extensions.js';

export const IGPUResourceService = createDecorator<IGPUResourceService>('gpuResourceService');

export interface GPUInfo {
	readonly id: number;
	readonly name: string;
	readonly driverVersion: string;
	readonly memoryTotal: number; // in MB
	readonly computeCapability: string;
	readonly temperature: number; // in Celsius
	readonly powerDraw: number; // in Watts
	readonly powerLimit: number; // in Watts
}

export interface GPUUtilization {
	readonly gpuId: number;
	readonly gpuUtilization: number; // percentage 0-100
	readonly memoryUtilization: number; // percentage 0-100
	readonly memoryUsed: number; // in MB
	readonly memoryFree: number; // in MB
	readonly temperature: number; // in Celsius
	readonly powerDraw: number; // in Watts
	readonly processes: GPUProcess[];
}

export interface GPUProcess {
	readonly pid: number;
	readonly name: string;
	readonly memoryUsage: number; // in MB
	readonly gpuUtilization: number; // percentage 0-100
	readonly type: 'C' | 'G' | 'C+G'; // Compute, Graphics, or Compute+Graphics
}

export interface TPUInfo {
	readonly name: string;
	readonly version: string;
	readonly memoryTotal: number; // in GB
	readonly acceleratorType: string;
}

export interface TPUUtilization {
	readonly name: string;
	readonly utilization: number; // percentage 0-100
	readonly memoryUtilization: number; // percentage 0-100
	readonly temperature: number; // in Celsius
}

export interface ResourceAlert {
	readonly type: 'memory' | 'temperature' | 'power';
	readonly severity: 'warning' | 'critical';
	readonly message: string;
	readonly gpuId?: number;
	readonly threshold: number;
	readonly current: number;
	readonly timestamp: Date;
}

export interface IResourceThresholds {
	memoryWarning: number; // percentage
	memoryCritical: number; // percentage
	temperatureWarning: number; // Celsius
	temperatureCritical: number; // Celsius
	powerWarning: number; // percentage of power limit
}

export interface IResourceMonitoringState {
	readonly gpus: GPUInfo[];
	readonly tpus: TPUInfo[];
	readonly gpuUtilization: Map<number, GPUUtilization>;
	readonly tpuUtilization: Map<string, TPUUtilization>;
	readonly alerts: ResourceAlert[];
	readonly isMonitoring: boolean;
	readonly lastUpdate: Date;
}

export interface IGPUResourceService {
	readonly _serviceBrand: undefined;

	readonly onDidUpdateGPUUtilization: Event<Map<number, GPUUtilization>>;
	readonly onDidUpdateTPUUtilization: Event<Map<string, TPUUtilization>>;
	readonly onDidTriggerAlert: Event<ResourceAlert>;
	readonly onDidUpdateMonitoringState: Event<IResourceMonitoringState>;

	readonly state: IResourceMonitoringState;

	startMonitoring(): Promise<void>;
	stopMonitoring(): void;
	getGPUInfo(): Promise<GPUInfo[]>;
	getTPUInfo(): Promise<TPUInfo[]>;
	getGPUUtilization(gpuId?: number): Promise<GPUUtilization[]>;
	getTPUUtilization(tpuName?: string): Promise<TPUUtilization[]>;
	setResourceThresholds(thresholds: Partial<IResourceThresholds>): void;
	killProcess(pid: number): Promise<boolean>;
	getHistoricalData(gpuId: number, duration: number): Promise<{ timestamp: Date; utilization: GPUUtilization }[]>;
}

export class GPUResourceService extends Disposable implements IGPUResourceService {
	declare readonly _serviceBrand: undefined;

	private readonly _onDidUpdateGPUUtilization = this._register(new Emitter<Map<number, GPUUtilization>>());
	public readonly onDidUpdateGPUUtilization = this._onDidUpdateGPUUtilization.event;

	private readonly _onDidUpdateTPUUtilization = this._register(new Emitter<Map<string, TPUUtilization>>());
	public readonly onDidUpdateTPUUtilization = this._onDidUpdateTPUUtilization.event;

	private readonly _onDidTriggerAlert = this._register(new Emitter<ResourceAlert>());
	public readonly onDidTriggerAlert = this._onDidTriggerAlert.event;

	private readonly _onDidUpdateMonitoringState = this._register(new Emitter<IResourceMonitoringState>());
	public readonly onDidUpdateMonitoringState = this._onDidUpdateMonitoringState.event;

	private _state: IResourceMonitoringState;
	private _monitoringInterval?: NodeJS.Timeout;
	private _thresholds: IResourceThresholds;
	private _historicalData: Map<number, { timestamp: Date; utilization: GPUUtilization }[]>;

	constructor() {
		super();

		this._thresholds = {
			memoryWarning: 80,
			memoryCritical: 95,
			temperatureWarning: 80,
			temperatureCritical: 90,
			powerWarning: 90
		};

		this._historicalData = new Map();

		this._state = {
			gpus: [],
			tpus: [],
			gpuUtilization: new Map(),
			tpuUtilization: new Map(),
			alerts: [],
			isMonitoring: false,
			lastUpdate: new Date()
		};
	}

	get state(): IResourceMonitoringState {
		return this._state;
	}

	async startMonitoring(): Promise<void> {
		if (this._state.isMonitoring) {
			return;
		}

		// Initialize GPU and TPU info
		const [gpus, tpus] = await Promise.all([
			this.getGPUInfo(),
			this.getTPUInfo()
		]);

		this._state = {
			...this._state,
			gpus,
			tpus,
			isMonitoring: true
		};

		// Start periodic monitoring
		this._monitoringInterval = setInterval(async () => {
			await this._updateUtilization();
		}, 2000); // Update every 2 seconds

		this._register(toDisposable(() => {
			if (this._monitoringInterval) {
				clearInterval(this._monitoringInterval);
			}
		}));

		this._onDidUpdateMonitoringState.fire(this._state);
	}

	stopMonitoring(): void {
		if (this._monitoringInterval) {
			clearInterval(this._monitoringInterval);
			this._monitoringInterval = undefined;
		}

		this._state = {
			...this._state,
			isMonitoring: false
		};

		this._onDidUpdateMonitoringState.fire(this._state);
	}

	async getGPUInfo(): Promise<GPUInfo[]> {
		try {
			// This would typically use nvidia-ml-py or nvidia-smi
			// For now, returning mock data
			return [
				{
					id: 0,
					name: 'NVIDIA GeForce RTX 4090',
					driverVersion: '535.98',
					memoryTotal: 24576, // 24GB
					computeCapability: '8.9',
					temperature: 45,
					powerDraw: 150,
					powerLimit: 450
				},
				{
					id: 1,
					name: 'NVIDIA GeForce RTX 3080',
					driverVersion: '535.98',
					memoryTotal: 10240, // 10GB
					computeCapability: '8.6',
					temperature: 52,
					powerDraw: 180,
					powerLimit: 320
				}
			];
		} catch (error) {
			console.error('Failed to get GPU info:', error);
			return [];
		}
	}

	async getTPUInfo(): Promise<TPUInfo[]> {
		try {
			// This would typically use Google Cloud TPU API
			// For now, returning mock data
			return [
				{
					name: 'tpu-v3-8',
					version: 'v3',
					memoryTotal: 128, // 128GB HBM
					acceleratorType: 'TPU v3'
				}
			];
		} catch (error) {
			console.error('Failed to get TPU info:', error);
			return [];
		}
	}

	async getGPUUtilization(gpuId?: number): Promise<GPUUtilization[]> {
		try {
			// Mock GPU utilization data
			const utilizations: GPUUtilization[] = [];

			for (const gpu of this._state.gpus) {
				if (gpuId !== undefined && gpu.id !== gpuId) {
					continue;
				}

				const memoryUsed = Math.floor(Math.random() * gpu.memoryTotal * 0.8);
				const utilization: GPUUtilization = {
					gpuId: gpu.id,
					gpuUtilization: Math.floor(Math.random() * 100),
					memoryUtilization: Math.floor((memoryUsed / gpu.memoryTotal) * 100),
					memoryUsed,
					memoryFree: gpu.memoryTotal - memoryUsed,
					temperature: gpu.temperature + Math.floor(Math.random() * 20) - 10,
					powerDraw: gpu.powerDraw + Math.floor(Math.random() * 100) - 50,
					processes: [
						{
							pid: 12345,
							name: 'python',
							memoryUsage: Math.floor(memoryUsed * 0.6),
							gpuUtilization: 45,
							type: 'C'
						},
						{
							pid: 67890,
							name: 'jupyter-lab',
							memoryUsage: Math.floor(memoryUsed * 0.4),
							gpuUtilization: 25,
							type: 'C'
						}
					]
				};

				utilizations.push(utilization);
			}

			return utilizations;
		} catch (error) {
			console.error('Failed to get GPU utilization:', error);
			return [];
		}
	}

	async getTPUUtilization(tpuName?: string): Promise<TPUUtilization[]> {
		try {
			// Mock TPU utilization data
			const utilizations: TPUUtilization[] = [];

			for (const tpu of this._state.tpus) {
				if (tpuName !== undefined && tpu.name !== tpuName) {
					continue;
				}

				const utilization: TPUUtilization = {
					name: tpu.name,
					utilization: Math.floor(Math.random() * 100),
					memoryUtilization: Math.floor(Math.random() * 80),
					temperature: 65 + Math.floor(Math.random() * 20)
				};

				utilizations.push(utilization);
			}

			return utilizations;
		} catch (error) {
			console.error('Failed to get TPU utilization:', error);
			return [];
		}
	}

	setResourceThresholds(thresholds: Partial<IResourceThresholds>): void {
		this._thresholds = { ...this._thresholds, ...thresholds };
	}

	async killProcess(pid: number): Promise<boolean> {
		try {
			// This would typically use process.kill or system commands
			console.log(`Attempting to kill process ${pid}`);
			return true;
		} catch (error) {
			console.error(`Failed to kill process ${pid}:`, error);
			return false;
		}
	}

	async getHistoricalData(gpuId: number, duration: number): Promise<{ timestamp: Date; utilization: GPUUtilization }[]> {
		const data = this._historicalData.get(gpuId) || [];
		const cutoff = new Date(Date.now() - duration * 1000);
		return data.filter(entry => entry.timestamp > cutoff);
	}

	private async _updateUtilization(): Promise<void> {
		try {
			const [gpuUtilizations, tpuUtilizations] = await Promise.all([
				this.getGPUUtilization(),
				this.getTPUUtilization()
			]);

			// Update GPU utilization
			const gpuUtilizationMap = new Map<number, GPUUtilization>();
			for (const util of gpuUtilizations) {
				gpuUtilizationMap.set(util.gpuId, util);

				// Store historical data
				if (!this._historicalData.has(util.gpuId)) {
					this._historicalData.set(util.gpuId, []);
				}
				const history = this._historicalData.get(util.gpuId)!;
				history.push({ timestamp: new Date(), utilization: util });

				// Keep only last 1 hour of data
				const cutoff = new Date(Date.now() - 3600 * 1000);
				this._historicalData.set(util.gpuId, history.filter(entry => entry.timestamp > cutoff));

				// Check for alerts
				this._checkAlerts(util);
			}

			// Update TPU utilization
			const tpuUtilizationMap = new Map<string, TPUUtilization>();
			for (const util of tpuUtilizations) {
				tpuUtilizationMap.set(util.name, util);
			}

			this._state = {
				...this._state,
				gpuUtilization: gpuUtilizationMap,
				tpuUtilization: tpuUtilizationMap,
				lastUpdate: new Date()
			};

			this._onDidUpdateGPUUtilization.fire(gpuUtilizationMap);
			this._onDidUpdateTPUUtilization.fire(tpuUtilizationMap);
			this._onDidUpdateMonitoringState.fire(this._state);

		} catch (error) {
			console.error('Failed to update utilization:', error);
		}
	}

	private _checkAlerts(utilization: GPUUtilization): void {
		const alerts: ResourceAlert[] = [];

		// Check memory alerts
		if (utilization.memoryUtilization >= this._thresholds.memoryCritical) {
			alerts.push({
				type: 'memory',
				severity: 'critical',
				message: `GPU ${utilization.gpuId} memory usage is critically high`,
				gpuId: utilization.gpuId,
				threshold: this._thresholds.memoryCritical,
				current: utilization.memoryUtilization,
				timestamp: new Date()
			});
		} else if (utilization.memoryUtilization >= this._thresholds.memoryWarning) {
			alerts.push({
				type: 'memory',
				severity: 'warning',
				message: `GPU ${utilization.gpuId} memory usage is high`,
				gpuId: utilization.gpuId,
				threshold: this._thresholds.memoryWarning,
				current: utilization.memoryUtilization,
				timestamp: new Date()
			});
		}

		// Check temperature alerts
		if (utilization.temperature >= this._thresholds.temperatureCritical) {
			alerts.push({
				type: 'temperature',
				severity: 'critical',
				message: `GPU ${utilization.gpuId} temperature is critically high`,
				gpuId: utilization.gpuId,
				threshold: this._thresholds.temperatureCritical,
				current: utilization.temperature,
				timestamp: new Date()
			});
		} else if (utilization.temperature >= this._thresholds.temperatureWarning) {
			alerts.push({
				type: 'temperature',
				severity: 'warning',
				message: `GPU ${utilization.gpuId} temperature is high`,
				gpuId: utilization.gpuId,
				threshold: this._thresholds.temperatureWarning,
				current: utilization.temperature,
				timestamp: new Date()
			});
		}

		// Check power alerts
		const gpu = this._state.gpus.find(g => g.id === utilization.gpuId);
		if (gpu) {
			const powerPercentage = (utilization.powerDraw / gpu.powerLimit) * 100;
			if (powerPercentage >= this._thresholds.powerWarning) {
				alerts.push({
					type: 'power',
					severity: 'warning',
					message: `GPU ${utilization.gpuId} power draw is high`,
					gpuId: utilization.gpuId,
					threshold: this._thresholds.powerWarning,
					current: powerPercentage,
					timestamp: new Date()
				});
			}
		}

		// Fire alerts
		for (const alert of alerts) {
			// Add to state alerts (keep only recent alerts)
			const updatedAlerts = [...this._state.alerts, alert]
				.filter(a => Date.now() - a.timestamp.getTime() < 300000) // Keep alerts for 5 minutes
				.slice(-10); // Keep only last 10 alerts

			// Update state with new alerts
			this._state = {
				...this._state,
				alerts: updatedAlerts
			};

			this._onDidTriggerAlert.fire(alert);
		}
	}
}

registerSingleton(IGPUResourceService, GPUResourceService, InstantiationType.Delayed);
