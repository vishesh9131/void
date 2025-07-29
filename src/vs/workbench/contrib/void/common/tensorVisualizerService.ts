/*--------------------------------------------------------------------------------------
 *  Copyright 2025 VS Aware Development Team. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

import { Event, Emitter } from '../../../../base/common/event.js';
import { Disposable } from '../../../../base/common/lifecycle.js';
import { createDecorator } from '../../../../platform/instantiation/common/instantiation.js';
import { registerSingleton, InstantiationType } from '../../../../platform/instantiation/common/extensions.js';
import { Position } from '../../../../editor/common/core/position.js';

export const ITensorVisualizerService = createDecorator<ITensorVisualizerService>('tensorVisualizerService');

export interface TensorInfo {
	readonly name: string;
	readonly shape: number[];
	readonly dtype: string;
	readonly size: number;
	readonly framework: 'numpy' | 'torch' | 'tensorflow' | 'pandas';
	readonly memoryUsage: number; // in bytes
	readonly device?: string; // for torch/tf tensors
}

export interface TensorStatistics {
	readonly mean: number;
	readonly std: number;
	readonly min: number;
	readonly max: number;
	readonly median: number;
	readonly q25: number;
	readonly q75: number;
	readonly nanCount: number;
	readonly infCount: number;
	readonly zeros: number;
	readonly nonZeros: number;
	readonly sparsity: number; // percentage of zeros
}

export interface TensorVisualization {
	readonly type: 'heatmap' | 'histogram' | 'line' | 'scatter' | 'image' | 'text';
	readonly data: any;
	readonly config: VisualizationConfig;
}

export interface VisualizationConfig {
	readonly width?: number;
	readonly height?: number;
	readonly colormap?: string;
	readonly title?: string;
	readonly xLabel?: string;
	readonly yLabel?: string;
	readonly showValues?: boolean;
	readonly interactive?: boolean;
}

export interface DataFrameInfo {
	readonly name: string;
	readonly shape: [number, number]; // [rows, columns]
	readonly columns: DataFrameColumn[];
	readonly dtypes: Record<string, string>;
	readonly memoryUsage: number;
	readonly index: DataFrameIndex;
}

export interface DataFrameColumn {
	readonly name: string;
	readonly dtype: string;
	readonly nullCount: number;
	readonly uniqueCount: number;
	readonly memoryUsage: number;
}

export interface DataFrameIndex {
	readonly name?: string;
	readonly dtype: string;
	readonly length: number;
	readonly isUnique: boolean;
}

export interface DataFrameStatistics {
	readonly numerical: Record<string, TensorStatistics>;
	readonly categorical: Record<string, CategoricalStatistics>;
	readonly overview: DataFrameOverview;
}

export interface CategoricalStatistics {
	readonly uniqueCount: number;
	readonly topValue: string;
	readonly topCount: number;
	readonly frequency: Record<string, number>;
}

export interface DataFrameOverview {
	readonly duplicateRows: number;
	readonly missingValuesByColumn: Record<string, number>;
	readonly correlationMatrix?: number[][];
	readonly correlationLabels?: string[];
}

export interface DimensionalityReduction {
	readonly method: 'pca' | 'tsne' | 'umap';
	readonly originalDimensions: number;
	readonly reducedDimensions: number;
	readonly data: number[][];
	readonly explainedVariance?: number[];
	readonly labels?: string[];
}

export interface HoverInfo {
	readonly position: Position;
	readonly variableName: string;
	readonly variableType: 'tensor' | 'dataframe' | 'array';
	readonly tensorInfo?: TensorInfo;
	readonly dataFrameInfo?: DataFrameInfo;
	readonly previewData?: any;
	readonly quickStats?: TensorStatistics | DataFrameStatistics;
}

export interface ITensorVisualizerService {
	readonly _serviceBrand: undefined;

	readonly onDidUpdateTensorInfo: Event<TensorInfo>;
	readonly onDidUpdateDataFrameInfo: Event<DataFrameInfo>;
	readonly onDidGenerateVisualization: Event<TensorVisualization>;

	// Tensor operations
	inspectTensor(code: string, variableName: string): Promise<TensorInfo | undefined>;
	getTensorStatistics(tensorInfo: TensorInfo): Promise<TensorStatistics>;
	generateTensorVisualization(tensorInfo: TensorInfo, type: TensorVisualization['type']): Promise<TensorVisualization>;
	sliceTensor(tensorInfo: TensorInfo, sliceExpression: string): Promise<TensorInfo>;

	// DataFrame operations
	inspectDataFrame(code: string, variableName: string): Promise<DataFrameInfo | undefined>;
	getDataFrameStatistics(dataFrameInfo: DataFrameInfo): Promise<DataFrameStatistics>;
	generateDataFrameVisualization(dataFrameInfo: DataFrameInfo, column?: string, type?: string): Promise<TensorVisualization>;
	queryDataFrame(dataFrameInfo: DataFrameInfo, query: string): Promise<DataFrameInfo>;

	// Dimensionality reduction
	reduceDimensionality(data: number[][], method: DimensionalityReduction['method'], targetDimensions: number): Promise<DimensionalityReduction>;

	// Hover functionality
	getHoverInfo(code: string, position: Position): Promise<HoverInfo | undefined>;
	
	// Code execution for inspection
	executeInspectionCode(code: string): Promise<any>;
}

export class TensorVisualizerService extends Disposable implements ITensorVisualizerService {
	declare readonly _serviceBrand: undefined;

	private readonly _onDidUpdateTensorInfo = this._register(new Emitter<TensorInfo>());
	public readonly onDidUpdateTensorInfo = this._onDidUpdateTensorInfo.event;

	private readonly _onDidUpdateDataFrameInfo = this._register(new Emitter<DataFrameInfo>());
	public readonly onDidUpdateDataFrameInfo = this._onDidUpdateDataFrameInfo.event;

	private readonly _onDidGenerateVisualization = this._register(new Emitter<TensorVisualization>());
	public readonly onDidGenerateVisualization = this._onDidGenerateVisualization.event;

	constructor() {
		super();
	}

	async inspectTensor(code: string, variableName: string): Promise<TensorInfo | undefined> {
		try {
			// This would typically execute the code and inspect the tensor
			// For now, returning mock data
			const tensorInfo: TensorInfo = {
				name: variableName,
				shape: [128, 256, 3],
				dtype: 'float32',
				size: 128 * 256 * 3,
				framework: 'torch',
				memoryUsage: 128 * 256 * 3 * 4, // 4 bytes per float32
				device: 'cuda:0'
			};

			this._onDidUpdateTensorInfo.fire(tensorInfo);
			return tensorInfo;
		} catch (error) {
			console.error('Failed to inspect tensor:', error);
			return undefined;
		}
	}

	async getTensorStatistics(tensorInfo: TensorInfo): Promise<TensorStatistics> {
		// Mock statistics calculation
		return {
			mean: 0.5,
			std: 0.25,
			min: 0.0,
			max: 1.0,
			median: 0.48,
			q25: 0.25,
			q75: 0.75,
			nanCount: 0,
			infCount: 0,
			zeros: 1250,
			nonZeros: 98302,
			sparsity: 1.26
		};
	}

	async generateTensorVisualization(tensorInfo: TensorInfo, type: TensorVisualization['type']): Promise<TensorVisualization> {
		const visualization: TensorVisualization = {
			type,
			data: this._generateMockVisualizationData(tensorInfo, type),
			config: {
				width: 800,
				height: 600,
				title: `${tensorInfo.name} (${tensorInfo.shape.join('×')})`,
				interactive: true
			}
		};

		this._onDidGenerateVisualization.fire(visualization);
		return visualization;
	}

	async sliceTensor(tensorInfo: TensorInfo, sliceExpression: string): Promise<TensorInfo> {
		// Mock tensor slicing
		const slicedShape = tensorInfo.shape.slice(); // Simplified
		return {
			...tensorInfo,
			name: `${tensorInfo.name}[${sliceExpression}]`,
			shape: slicedShape,
			size: slicedShape.reduce((a, b) => a * b, 1)
		};
	}

	async inspectDataFrame(code: string, variableName: string): Promise<DataFrameInfo | undefined> {
		try {
			// Mock DataFrame inspection
			const dataFrameInfo: DataFrameInfo = {
				name: variableName,
				shape: [10000, 15],
				columns: [
					{ name: 'feature_1', dtype: 'float64', nullCount: 0, uniqueCount: 9850, memoryUsage: 80000 },
					{ name: 'feature_2', dtype: 'int64', nullCount: 5, uniqueCount: 100, memoryUsage: 80000 },
					{ name: 'category', dtype: 'object', nullCount: 0, uniqueCount: 5, memoryUsage: 640000 },
					{ name: 'target', dtype: 'int64', nullCount: 0, uniqueCount: 2, memoryUsage: 80000 }
				],
				dtypes: {
					'feature_1': 'float64',
					'feature_2': 'int64',
					'category': 'object',
					'target': 'int64'
				},
				memoryUsage: 880000,
				index: {
					name: undefined,
					dtype: 'int64',
					length: 10000,
					isUnique: true
				}
			};

			this._onDidUpdateDataFrameInfo.fire(dataFrameInfo);
			return dataFrameInfo;
		} catch (error) {
			console.error('Failed to inspect DataFrame:', error);
			return undefined;
		}
	}

	async getDataFrameStatistics(dataFrameInfo: DataFrameInfo): Promise<DataFrameStatistics> {
		return {
			numerical: {
				'feature_1': {
					mean: 0.5,
					std: 0.3,
					min: -2.5,
					max: 3.2,
					median: 0.48,
					q25: 0.2,
					q75: 0.8,
					nanCount: 0,
					infCount: 0,
					zeros: 25,
					nonZeros: 9975,
					sparsity: 0.25
				},
				'feature_2': {
					mean: 50.2,
					std: 15.8,
					min: 10,
					max: 100,
					median: 48,
					q25: 38,
					q75: 62,
					nanCount: 5,
					infCount: 0,
					zeros: 0,
					nonZeros: 9995,
					sparsity: 0
				}
			},
			categorical: {
				'category': {
					uniqueCount: 5,
					topValue: 'A',
					topCount: 3500,
					frequency: {
						'A': 3500,
						'B': 2800,
						'C': 2200,
						'D': 1000,
						'E': 500
					}
				}
			},
			overview: {
				duplicateRows: 15,
				missingValuesByColumn: {
					'feature_1': 0,
					'feature_2': 5,
					'category': 0,
					'target': 0
				},
				correlationMatrix: [
					[1.0, 0.3, 0.1, 0.8],
					[0.3, 1.0, 0.05, 0.4],
					[0.1, 0.05, 1.0, 0.2],
					[0.8, 0.4, 0.2, 1.0]
				],
				correlationLabels: ['feature_1', 'feature_2', 'category', 'target']
			}
		};
	}

	async generateDataFrameVisualization(dataFrameInfo: DataFrameInfo, column?: string, type?: string): Promise<TensorVisualization> {
		const visualizationType = type as TensorVisualization['type'] || 'histogram';
		
		const visualization: TensorVisualization = {
			type: visualizationType,
			data: this._generateMockDataFrameVisualization(dataFrameInfo, column, visualizationType),
			config: {
				width: 800,
				height: 600,
				title: column ? `${dataFrameInfo.name}['${column}']` : `${dataFrameInfo.name} Overview`,
				interactive: true
			}
		};

		this._onDidGenerateVisualization.fire(visualization);
		return visualization;
	}

	async queryDataFrame(dataFrameInfo: DataFrameInfo, query: string): Promise<DataFrameInfo> {
		// Mock DataFrame query result
		return {
			...dataFrameInfo,
			name: `${dataFrameInfo.name}.query('${query}')`,
			shape: [Math.floor(dataFrameInfo.shape[0] * 0.7), dataFrameInfo.shape[1]] // Assume 70% of rows match
		};
	}

	async reduceDimensionality(data: number[][], method: DimensionalityReduction['method'], targetDimensions: number): Promise<DimensionalityReduction> {
		// Mock dimensionality reduction
		const originalDimensions = data[0]?.length || 0;
		const reducedData = data.map(() => 
			Array.from({ length: targetDimensions }, () => Math.random() * 10 - 5)
		);

		const reduction: DimensionalityReduction = {
			method,
			originalDimensions,
			reducedDimensions: targetDimensions,
			data: reducedData,
			explainedVariance: method === 'pca' ? [0.45, 0.25, 0.15, 0.10, 0.05] : undefined,
			labels: data.map((_, i) => `Point ${i}`)
		};

		return reduction;
	}

	async getHoverInfo(code: string, position: Position): Promise<HoverInfo | undefined> {
		// Extract variable name at position (simplified)
		const lines = code.split('\n');
		const line = lines[position.lineNumber - 1];
		const word = this._extractWordAtPosition(line, position.column);

		if (!word) {
			return undefined;
		}

		// Mock hover info
		if (word.includes('tensor') || word.includes('torch') || word.includes('tf.')) {
			const tensorInfo = await this.inspectTensor(code, word);
			if (tensorInfo) {
				const quickStats = await this.getTensorStatistics(tensorInfo);
				return {
					position,
					variableName: word,
					variableType: 'tensor',
					tensorInfo,
					quickStats
				};
			}
		} else if (word.includes('df') || word.includes('data')) {
			const dataFrameInfo = await this.inspectDataFrame(code, word);
			if (dataFrameInfo) {
				const quickStats = await this.getDataFrameStatistics(dataFrameInfo);
				return {
					position,
					variableName: word,
					variableType: 'dataframe',
					dataFrameInfo,
					quickStats
				};
			}
		}

		return undefined;
	}

	async executeInspectionCode(code: string): Promise<any> {
		try {
			// This would typically execute code in a Python kernel
			// For now, returning mock result
			return {
				success: true,
				result: 'Mock execution result',
				variables: ['x', 'y', 'model', 'data']
			};
		} catch (error) {
			console.error('Failed to execute inspection code:', error);
			throw error;
		}
	}

	private _extractWordAtPosition(line: string, column: number): string {
		const start = Math.max(0, line.lastIndexOf(' ', column - 1) + 1);
		const end = line.indexOf(' ', column);
		return line.substring(start, end === -1 ? line.length : end);
	}

	private _generateMockVisualizationData(tensorInfo: TensorInfo, type: TensorVisualization['type']): any {
		switch (type) {
			case 'histogram':
				return {
					bins: Array.from({ length: 50 }, (_, i) => i * 0.04),
					counts: Array.from({ length: 50 }, () => Math.floor(Math.random() * 1000))
				};
			case 'heatmap':
				const size = Math.min(tensorInfo.shape[0] || 32, 32);
				return Array.from({ length: size }, () => 
					Array.from({ length: size }, () => Math.random())
				);
			case 'line':
				return {
					x: Array.from({ length: 100 }, (_, i) => i),
					y: Array.from({ length: 100 }, () => Math.sin(Math.random() * Math.PI * 2))
				};
			case 'image':
				return {
					width: tensorInfo.shape[1] || 256,
					height: tensorInfo.shape[0] || 256,
					channels: tensorInfo.shape[2] || 3,
					data: 'base64encodedimagedata...'
				};
			default:
				return {};
		}
	}

	private _generateMockDataFrameVisualization(dataFrameInfo: DataFrameInfo, column?: string, type?: string): any {
		switch (type) {
			case 'histogram':
				return {
					bins: Array.from({ length: 20 }, (_, i) => i * 5),
					counts: Array.from({ length: 20 }, () => Math.floor(Math.random() * 500))
				};
			case 'scatter':
				return {
					x: Array.from({ length: 1000 }, () => Math.random() * 100),
					y: Array.from({ length: 1000 }, () => Math.random() * 100)
				};
			case 'heatmap':
				// Correlation matrix
				const numCols = dataFrameInfo.columns.length;
				return Array.from({ length: numCols }, () => 
					Array.from({ length: numCols }, () => Math.random() * 2 - 1)
				);
			default:
				return {};
		}
	}
}

registerSingleton(ITensorVisualizerService, TensorVisualizerService, InstantiationType.Delayed);