/*--------------------------------------------------------------------------------------
 *  Copyright 2025 VS Aware Development Team. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

import { Event, Emitter } from '../../../../base/common/event.js';
import { Disposable } from '../../../../base/common/lifecycle.js';
import { createDecorator } from '../../../../platform/instantiation/common/instantiation.js';
import { registerSingleton, InstantiationType } from '../../../../platform/instantiation/common/extensions.js';
import { URI } from '../../../../base/common/uri.js';

export const IMLVersionControlService = createDecorator<IMLVersionControlService>('mlVersionControlService');

export interface ModelVersion {
	readonly id: string;
	readonly name: string;
	readonly version: string;
	readonly path: string;
	readonly size: number;
	readonly framework: 'pytorch' | 'tensorflow' | 'sklearn' | 'onnx' | 'other';
	readonly architecture: any;
	readonly metadata: ModelMetadata;
	readonly createdAt: Date;
	readonly commitHash?: string;
	readonly parentVersion?: string;
}

export interface ModelMetadata {
	readonly inputShape?: number[];
	readonly outputShape?: number[];
	readonly parameters: number;
	readonly hyperparameters: Record<string, any>;
	readonly trainingDataset?: string;
	readonly validationMetrics: Record<string, number>;
	readonly tags: string[];
	readonly description?: string;
}

export interface DatasetVersion {
	readonly id: string;
	readonly name: string;
	readonly version: string;
	readonly path: string;
	readonly size: number;
	readonly recordCount: number;
	readonly schema: DatasetSchema;
	readonly statistics: DatasetStatistics;
	readonly createdAt: Date;
	readonly commitHash?: string;
	readonly parentVersion?: string;
}

export interface DatasetSchema {
	readonly columns: DatasetColumn[];
	readonly primaryKey?: string;
	readonly targetColumn?: string;
}

export interface DatasetColumn {
	readonly name: string;
	readonly dataType: 'int64' | 'float64' | 'object' | 'bool' | 'datetime';
	readonly nullable: boolean;
	readonly unique: boolean;
}

export interface DatasetStatistics {
	readonly nullCount: Record<string, number>;
	readonly uniqueCount: Record<string, number>;
	readonly mean: Record<string, number>;
	readonly std: Record<string, number>;
	readonly min: Record<string, number>;
	readonly max: Record<string, number>;
	readonly distribution: Record<string, number[]>;
}

export interface ModelDiff {
	readonly oldVersion: ModelVersion;
	readonly newVersion: ModelVersion;
	readonly weightChanges: WeightChange[];
	readonly architectureChanges: ArchitectureChange[];
	readonly metadataChanges: MetadataChange[];
	readonly performanceChanges: PerformanceChange[];
}

export interface WeightChange {
	readonly layerName: string;
	readonly parameterName: string;
	readonly changeType: 'added' | 'removed' | 'modified';
	readonly oldShape?: number[];
	readonly newShape?: number[];
	readonly magnitudeChange?: number;
	readonly histogram?: WeightHistogram;
}

export interface WeightHistogram {
	readonly bins: number[];
	readonly counts: number[];
	readonly oldCounts?: number[];
}

export interface ArchitectureChange {
	readonly changeType: 'layer_added' | 'layer_removed' | 'layer_modified';
	readonly layerName: string;
	readonly oldConfig?: any;
	readonly newConfig?: any;
}

export interface MetadataChange {
	readonly field: string;
	readonly oldValue: any;
	readonly newValue: any;
}

export interface PerformanceChange {
	readonly metric: string;
	readonly oldValue: number;
	readonly newValue: number;
	readonly changePercent: number;
	readonly improved: boolean;
}

export interface DatasetDiff {
	readonly oldVersion: DatasetVersion;
	readonly newVersion: DatasetVersion;
	readonly schemaChanges: SchemaChange[];
	readonly statisticalChanges: StatisticalChange[];
	readonly recordChanges: RecordChange;
}

export interface SchemaChange {
	readonly changeType: 'column_added' | 'column_removed' | 'column_modified';
	readonly columnName: string;
	readonly oldColumn?: DatasetColumn;
	readonly newColumn?: DatasetColumn;
}

export interface StatisticalChange {
	readonly columnName: string;
	readonly statistic: string;
	readonly oldValue: number;
	readonly newValue: number;
	readonly changePercent: number;
	readonly significant: boolean;
}

export interface RecordChange {
	readonly added: number;
	readonly removed: number;
	readonly modified: number;
	readonly total: number;
}

export interface ExperimentConfig {
	readonly id: string;
	readonly name: string;
	readonly modelVersion: string;
	readonly datasetVersion: string;
	readonly hyperparameters: Record<string, any>;
	readonly environment: Record<string, string>;
	readonly createdAt: Date;
}

export interface IMLVersionControlService {
	readonly _serviceBrand: undefined;

	readonly onDidAddModelVersion: Event<ModelVersion>;
	readonly onDidAddDatasetVersion: Event<DatasetVersion>;
	readonly onDidUpdateVersion: Event<ModelVersion | DatasetVersion>;

	// Model version control
	addModelVersion(model: Omit<ModelVersion, 'id' | 'createdAt'>): Promise<ModelVersion>;
	getModelVersions(modelName?: string): Promise<ModelVersion[]>;
	getModelVersion(id: string): Promise<ModelVersion | undefined>;
	compareModelVersions(oldId: string, newId: string): Promise<ModelDiff>;
	deleteModelVersion(id: string): Promise<boolean>;
	tagModelVersion(id: string, tag: string): Promise<void>;
	
	// Dataset version control
	addDatasetVersion(dataset: Omit<DatasetVersion, 'id' | 'createdAt'>): Promise<DatasetVersion>;
	getDatasetVersions(datasetName?: string): Promise<DatasetVersion[]>;
	getDatasetVersion(id: string): Promise<DatasetVersion | undefined>;
	compareDatasetVersions(oldId: string, newId: string): Promise<DatasetDiff>;
	deleteDatasetVersion(id: string): Promise<boolean>;
	tagDatasetVersion(id: string, tag: string): Promise<void>;

	// Experiment tracking
	createExperiment(config: Omit<ExperimentConfig, 'id' | 'createdAt'>): Promise<ExperimentConfig>;
	getExperiments(): Promise<ExperimentConfig[]>;
	
	// DVC/Git-LFS integration
	initializeDVC(workspaceUri: URI): Promise<boolean>;
	addToGitLFS(uri: URI): Promise<boolean>;
	pushToRemote(): Promise<boolean>;
	pullFromRemote(): Promise<boolean>;
	getRemoteStatus(): Promise<{ ahead: number; behind: number; dirty: boolean }>;
}

export class MLVersionControlService extends Disposable implements IMLVersionControlService {
	declare readonly _serviceBrand: undefined;

	private readonly _onDidAddModelVersion = this._register(new Emitter<ModelVersion>());
	public readonly onDidAddModelVersion = this._onDidAddModelVersion.event;

	private readonly _onDidAddDatasetVersion = this._register(new Emitter<DatasetVersion>());
	public readonly onDidAddDatasetVersion = this._onDidAddDatasetVersion.event;

	private readonly _onDidUpdateVersion = this._register(new Emitter<ModelVersion | DatasetVersion>());
	public readonly onDidUpdateVersion = this._onDidUpdateVersion.event;

	private _modelVersions: Map<string, ModelVersion> = new Map();
	private _datasetVersions: Map<string, DatasetVersion> = new Map();
	private _experiments: Map<string, ExperimentConfig> = new Map();

	constructor() {
		super();
	}

	async addModelVersion(model: Omit<ModelVersion, 'id' | 'createdAt'>): Promise<ModelVersion> {
		const id = this._generateId();
		const modelVersion: ModelVersion = {
			...model,
			id,
			createdAt: new Date()
		};

		this._modelVersions.set(id, modelVersion);
		this._onDidAddModelVersion.fire(modelVersion);
		
		return modelVersion;
	}

	async getModelVersions(modelName?: string): Promise<ModelVersion[]> {
		const versions = Array.from(this._modelVersions.values());
		if (modelName) {
			return versions.filter(v => v.name === modelName);
		}
		return versions.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
	}

	async getModelVersion(id: string): Promise<ModelVersion | undefined> {
		return this._modelVersions.get(id);
	}

	async compareModelVersions(oldId: string, newId: string): Promise<ModelDiff> {
		const oldVersion = this._modelVersions.get(oldId);
		const newVersion = this._modelVersions.get(newId);

		if (!oldVersion || !newVersion) {
			throw new Error('Model version not found');
		}

		// Generate mock diff data
		const weightChanges: WeightChange[] = [
			{
				layerName: 'conv1',
				parameterName: 'weight',
				changeType: 'modified',
				oldShape: [64, 3, 7, 7],
				newShape: [64, 3, 7, 7],
				magnitudeChange: 0.15,
				histogram: {
					bins: [-0.5, -0.3, -0.1, 0.1, 0.3, 0.5],
					counts: [10, 25, 100, 95, 30, 8],
					oldCounts: [12, 28, 85, 110, 25, 5]
				}
			},
			{
				layerName: 'fc',
				parameterName: 'bias',
				changeType: 'modified',
				oldShape: [1000],
				newShape: [1000],
				magnitudeChange: 0.08
			}
		];

		const architectureChanges: ArchitectureChange[] = [];
		
		const metadataChanges: MetadataChange[] = [
			{
				field: 'hyperparameters.learning_rate',
				oldValue: 0.001,
				newValue: 0.0005
			}
		];

		const performanceChanges: PerformanceChange[] = [
			{
				metric: 'accuracy',
				oldValue: oldVersion.metadata.validationMetrics.accuracy || 0,
				newValue: newVersion.metadata.validationMetrics.accuracy || 0,
				changePercent: 5.2,
				improved: true
			}
		];

		return {
			oldVersion,
			newVersion,
			weightChanges,
			architectureChanges,
			metadataChanges,
			performanceChanges
		};
	}

	async deleteModelVersion(id: string): Promise<boolean> {
		return this._modelVersions.delete(id);
	}

	async tagModelVersion(id: string, tag: string): Promise<void> {
		const version = this._modelVersions.get(id);
		if (version) {
			const updatedVersion = {
				...version,
				metadata: {
					...version.metadata,
					tags: [...version.metadata.tags, tag]
				}
			};
			this._modelVersions.set(id, updatedVersion);
			this._onDidUpdateVersion.fire(updatedVersion);
		}
	}

	async addDatasetVersion(dataset: Omit<DatasetVersion, 'id' | 'createdAt'>): Promise<DatasetVersion> {
		const id = this._generateId();
		const datasetVersion: DatasetVersion = {
			...dataset,
			id,
			createdAt: new Date()
		};

		this._datasetVersions.set(id, datasetVersion);
		this._onDidAddDatasetVersion.fire(datasetVersion);
		
		return datasetVersion;
	}

	async getDatasetVersions(datasetName?: string): Promise<DatasetVersion[]> {
		const versions = Array.from(this._datasetVersions.values());
		if (datasetName) {
			return versions.filter(v => v.name === datasetName);
		}
		return versions.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
	}

	async getDatasetVersion(id: string): Promise<DatasetVersion | undefined> {
		return this._datasetVersions.get(id);
	}

	async compareDatasetVersions(oldId: string, newId: string): Promise<DatasetDiff> {
		const oldVersion = this._datasetVersions.get(oldId);
		const newVersion = this._datasetVersions.get(newId);

		if (!oldVersion || !newVersion) {
			throw new Error('Dataset version not found');
		}

		// Generate mock diff data
		const schemaChanges: SchemaChange[] = [
			{
				changeType: 'column_added',
				columnName: 'new_feature',
				newColumn: {
					name: 'new_feature',
					dataType: 'float64',
					nullable: true,
					unique: false
				}
			}
		];

		const statisticalChanges: StatisticalChange[] = [
			{
				columnName: 'price',
				statistic: 'mean',
				oldValue: 100.5,
				newValue: 105.2,
				changePercent: 4.7,
				significant: true
			}
		];

		const recordChanges: RecordChange = {
			added: 1000,
			removed: 50,
			modified: 200,
			total: newVersion.recordCount
		};

		return {
			oldVersion,
			newVersion,
			schemaChanges,
			statisticalChanges,
			recordChanges
		};
	}

	async deleteDatasetVersion(id: string): Promise<boolean> {
		return this._datasetVersions.delete(id);
	}

	async tagDatasetVersion(id: string, tag: string): Promise<void> {
		const version = this._datasetVersions.get(id);
		if (version) {
			// Note: DatasetVersion doesn't have tags in its metadata structure
			// This would need to be extended or handled differently
			this._onDidUpdateVersion.fire(version);
		}
	}

	async createExperiment(config: Omit<ExperimentConfig, 'id' | 'createdAt'>): Promise<ExperimentConfig> {
		const id = this._generateId();
		const experiment: ExperimentConfig = {
			...config,
			id,
			createdAt: new Date()
		};

		this._experiments.set(id, experiment);
		return experiment;
	}

	async getExperiments(): Promise<ExperimentConfig[]> {
		return Array.from(this._experiments.values())
			.sort((a, b) => b.createdAt.getTime() - a.createdAt.getTime());
	}

	async initializeDVC(workspaceUri: URI): Promise<boolean> {
		try {
			// This would typically run `dvc init` command
			console.log(`Initializing DVC in workspace: ${workspaceUri.path}`);
			return true;
		} catch (error) {
			console.error('Failed to initialize DVC:', error);
			return false;
		}
	}

	async addToGitLFS(uri: URI): Promise<boolean> {
		try {
			// This would typically run `git lfs track` command
			console.log(`Adding to Git LFS: ${uri.path}`);
			return true;
		} catch (error) {
			console.error('Failed to add to Git LFS:', error);
			return false;
		}
	}

	async pushToRemote(): Promise<boolean> {
		try {
			// This would typically run `dvc push` and `git push` commands
			console.log('Pushing to remote repository');
			return true;
		} catch (error) {
			console.error('Failed to push to remote:', error);
			return false;
		}
	}

	async pullFromRemote(): Promise<boolean> {
		try {
			// This would typically run `dvc pull` and `git pull` commands
			console.log('Pulling from remote repository');
			return true;
		} catch (error) {
			console.error('Failed to pull from remote:', error);
			return false;
		}
	}

	async getRemoteStatus(): Promise<{ ahead: number; behind: number; dirty: boolean }> {
		try {
			// This would typically check git and DVC status
			return {
				ahead: 2,
				behind: 1,
				dirty: false
			};
		} catch (error) {
			console.error('Failed to get remote status:', error);
			return { ahead: 0, behind: 0, dirty: false };
		}
	}

	private _generateId(): string {
		return Math.random().toString(36).substr(2, 9) + Date.now().toString(36);
	}
}

registerSingleton(IMLVersionControlService, MLVersionControlService, InstantiationType.Delayed);