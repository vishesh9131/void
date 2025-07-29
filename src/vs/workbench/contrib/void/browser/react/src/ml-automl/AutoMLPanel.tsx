/*--------------------------------------------------------------------------------------
 *  Copyright 2025 VS Aware Development Team. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

import React, { useState, useEffect } from 'react';
import { Button, Card, CardContent, CardHeader, CardTitle } from '../util/components';
import { Upload, Brain, Code, Zap, BarChart3, Download } from 'lucide-react';

interface DatasetAnalysis {
	shape: [number, number];
	dataTypes: Record<string, string>;
	targetColumn?: string;
	problemType: 'classification' | 'regression' | 'clustering' | 'timeseries';
	missingValues: number;
	categoricalColumns: string[];
	numericalColumns: string[];
	statistics: Record<string, any>;
}

interface ModelRecommendation {
	framework: 'pytorch' | 'tensorflow' | 'sklearn';
	modelType: string;
	architecture: any;
	hyperparameters: Record<string, any>;
	reasoning: string;
	estimatedTrainingTime: string;
	expectedPerformance: number;
}

interface AutoMLResult {
	analysis: DatasetAnalysis;
	recommendations: ModelRecommendation[];
	generatedCode: string;
	requirements: string[];
}

interface AutoMLPanelProps {
	onAnalyzeDataset: (filePath: string) => Promise<AutoMLResult>;
	onGenerateCode: (framework: string, analysis: DatasetAnalysis) => Promise<string>;
}

const AutoMLPanel: React.FC<AutoMLPanelProps> = ({ onAnalyzeDataset, onGenerateCode }) => {
	const [selectedFile, setSelectedFile] = useState<string>('');
	const [analysis, setAnalysis] = useState<DatasetAnalysis | null>(null);
	const [recommendations, setRecommendations] = useState<ModelRecommendation[]>([]);
	const [generatedCode, setGeneratedCode] = useState<string>('');
	const [isAnalyzing, setIsAnalyzing] = useState(false);
	const [selectedFramework, setSelectedFramework] = useState<'pytorch' | 'tensorflow' | 'sklearn'>('sklearn');

	const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
		const file = event.target.files?.[0];
		if (file) {
			setSelectedFile(file.name);
		}
	};

	const handleAnalyzeDataset = async () => {
		if (!selectedFile) return;

		setIsAnalyzing(true);
		try {
			const result = await onAnalyzeDataset(selectedFile);
			setAnalysis(result.analysis);
			setRecommendations(result.recommendations);
			setGeneratedCode(result.generatedCode);
		} catch (error) {
			console.error('Failed to analyze dataset:', error);
		} finally {
			setIsAnalyzing(false);
		}
	};

	const handleGenerateCode = async (framework: string) => {
		if (!analysis) return;

		try {
			const code = await onGenerateCode(framework, analysis);
			setGeneratedCode(code);
			setSelectedFramework(framework as any);
		} catch (error) {
			console.error('Failed to generate code:', error);
		}
	};

	const downloadCode = () => {
		const element = document.createElement('a');
		const file = new Blob([generatedCode], { type: 'text/plain' });
		element.href = URL.createObjectURL(file);
		element.download = `model_${selectedFramework}.py`;
		document.body.appendChild(element);
		element.click();
		document.body.removeChild(element);
	};

	return (
		<div className="automl-panel p-4 space-y-6 max-w-6xl mx-auto">
			{/* Header */}
			<div className="text-center">
				<h1 className="text-3xl font-bold text-gray-900 dark:text-white mb-2">
					<Brain className="inline mr-2 h-8 w-8 text-blue-600" />
					AutoML Assistant
				</h1>
				<p className="text-gray-600 dark:text-gray-300">
					Analyze your dataset and generate optimized machine learning models
				</p>
			</div>

			{/* File Upload Section */}
			<Card>
				<CardHeader>
					<CardTitle className="flex items-center">
						<Upload className="mr-2 h-5 w-5" />
						Dataset Upload
					</CardTitle>
				</CardHeader>
				<CardContent>
					<div className="space-y-4">
						<div className="border-2 border-dashed border-gray-300 dark:border-gray-700 rounded-lg p-6 text-center">
							<input
								type="file"
								accept=".csv,.json,.parquet"
								onChange={handleFileUpload}
								className="hidden"
								id="file-upload"
							/>
							<label htmlFor="file-upload" className="cursor-pointer">
								<Upload className="mx-auto h-12 w-12 text-gray-400" />
								<p className="mt-2 text-sm text-gray-600 dark:text-gray-400">
									Click to upload CSV, JSON, or Parquet files
								</p>
								{selectedFile && (
									<p className="mt-1 text-sm text-blue-600 dark:text-blue-400">
										Selected: {selectedFile}
									</p>
								)}
							</label>
						</div>
						
						<Button
							onClick={handleAnalyzeDataset}
							disabled={!selectedFile || isAnalyzing}
							className="w-full"
						>
							{isAnalyzing ? (
								<>
									<Zap className="mr-2 h-4 w-4 animate-spin" />
									Analyzing Dataset...
								</>
							) : (
								<>
									<BarChart3 className="mr-2 h-4 w-4" />
									Analyze Dataset
								</>
							)}
						</Button>
					</div>
				</CardContent>
			</Card>

			{/* Analysis Results */}
			{analysis && (
				<div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
					{/* Dataset Overview */}
					<Card>
						<CardHeader>
							<CardTitle>Dataset Overview</CardTitle>
						</CardHeader>
						<CardContent>
							<div className="space-y-3">
								<div className="flex justify-between">
									<span className="font-medium">Shape:</span>
									<span>{analysis.shape[0]} × {analysis.shape[1]}</span>
								</div>
								<div className="flex justify-between">
									<span className="font-medium">Problem Type:</span>
									<span className="capitalize">{analysis.problemType}</span>
								</div>
								<div className="flex justify-between">
									<span className="font-medium">Missing Values:</span>
									<span>{analysis.missingValues}</span>
								</div>
								<div className="flex justify-between">
									<span className="font-medium">Target Column:</span>
									<span>{analysis.targetColumn || 'Not detected'}</span>
								</div>
								
								<div className="mt-4">
									<h4 className="font-medium mb-2">Column Types</h4>
									<div className="text-sm space-y-1">
										<div>Numerical: {analysis.numericalColumns.join(', ')}</div>
										<div>Categorical: {analysis.categoricalColumns.join(', ')}</div>
									</div>
								</div>
							</div>
						</CardContent>
					</Card>

					{/* Model Recommendations */}
					<Card>
						<CardHeader>
							<CardTitle>Model Recommendations</CardTitle>
						</CardHeader>
						<CardContent>
							<div className="space-y-4">
								{recommendations.map((rec, index) => (
									<div
										key={index}
										className="p-3 border rounded-lg dark:border-gray-700"
									>
										<div className="flex items-center justify-between mb-2">
											<h4 className="font-medium">{rec.modelType}</h4>
											<span className="text-sm bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 px-2 py-1 rounded">
												{rec.framework}
											</span>
										</div>
										<p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
											{rec.reasoning}
										</p>
										<div className="flex justify-between text-xs text-gray-500">
											<span>Training: {rec.estimatedTrainingTime}</span>
											<span>Expected: {(rec.expectedPerformance * 100).toFixed(1)}%</span>
										</div>
										<Button
											size="sm"
											onClick={() => handleGenerateCode(rec.framework)}
											className="mt-2 w-full"
											variant="outline"
										>
											Generate {rec.framework} Code
										</Button>
									</div>
								))}
							</div>
						</CardContent>
					</Card>
				</div>
			)}

			{/* Generated Code */}
			{generatedCode && (
				<Card>
					<CardHeader>
						<CardTitle className="flex items-center justify-between">
							<div className="flex items-center">
								<Code className="mr-2 h-5 w-5" />
								Generated {selectedFramework} Code
							</div>
							<Button onClick={downloadCode} size="sm" variant="outline">
								<Download className="mr-2 h-4 w-4" />
								Download
							</Button>
						</CardTitle>
					</CardHeader>
					<CardContent>
						<div className="relative">
							<pre className="bg-gray-100 dark:bg-gray-900 p-4 rounded-lg overflow-x-auto text-sm">
								<code>{generatedCode}</code>
							</pre>
						</div>
					</CardContent>
				</Card>
			)}

			{/* Tips and Help */}
			<Card>
				<CardHeader>
					<CardTitle>💡 Tips for Better Results</CardTitle>
				</CardHeader>
				<CardContent>
					<ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
						<li>• Ensure your dataset has a clear target column for supervised learning</li>
						<li>• Clean data (handle missing values) before uploading for better recommendations</li>
						<li>• For image datasets, consider uploading a CSV with file paths and labels</li>
						<li>• Larger datasets (>10K samples) will get deep learning recommendations</li>
						<li>• The generated code includes data preprocessing and model evaluation</li>
					</ul>
				</CardContent>
			</Card>
		</div>
	);
};

export default AutoMLPanel;