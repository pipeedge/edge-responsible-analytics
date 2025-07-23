import torch
import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Tuple, Union, Generator
import json
import time
from datetime import datetime
import gc
import os

from load_models import load_medgemma_model

logger = logging.getLogger(__name__)

class MedGemmaAnalyzer:
    """
    MedGemma-based medical analysis system optimized for Raspberry Pi 5 edge devices.
    Provides comprehensive medical text analysis, clinical decision support, and 
    diagnostic assistance using Google's Gemma model fine-tuned for medical tasks.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.medical_prompts = self._load_medical_prompts()
        
    def _load_medical_prompts(self) -> Dict[str, str]:
        """Load predefined medical analysis prompts"""
        return {
            "diagnosis": """You are a medical AI assistant. Based on the following medical information, provide a differential diagnosis with confidence levels:

Medical Information: {text}

Please provide:
1. Top 3 differential diagnoses with confidence percentages
2. Key clinical findings that support each diagnosis
3. Recommended next steps or tests
4. Risk stratification (Low/Medium/High)

Response format should be structured and concise.""",
            
            "symptom_analysis": """As a medical AI, analyze the following symptoms and patient presentation:

Patient Information: {text}

Please provide:
1. Symptom severity assessment
2. Potential underlying conditions
3. Red flag symptoms (if any)
4. Recommended clinical actions
5. Patient monitoring recommendations

Focus on evidence-based analysis.""",
            
            "drug_interaction": """Analyze potential drug interactions and medication safety:

Medication/Patient Information: {text}

Please assess:
1. Drug interaction risks
2. Contraindications
3. Dosage considerations
4. Patient-specific factors
5. Alternative medication suggestions if needed

Provide clear safety recommendations.""",
            
            "clinical_summary": """Summarize the following clinical case:

Clinical Information: {text}

Provide a concise clinical summary including:
1. Chief complaint and history
2. Key examination findings
3. Diagnostic impression
4. Treatment plan
5. Follow-up recommendations

Keep the summary professional and structured.""",
            
            "radiology_analysis": """Analyze the following radiology report or imaging findings:

Radiology Information: {text}

Please provide:
1. Key radiological findings
2. Clinical significance of findings
3. Differential diagnosis based on imaging
4. Correlation with clinical presentation
5. Follow-up imaging recommendations if needed

Focus on actionable insights."""
        }
    
    def load_model(self):
        """Load MedGemma model if not already loaded"""
        if not self.model_loaded:
            try:
                logger.info("Loading MedGemma model for edge inference...")
                self.model, self.tokenizer = load_medgemma_model()
                self.model_loaded = True
                logger.info("MedGemma model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load MedGemma model: {e}")
                raise
    
    def analyze_medical_text(self, text: str, analysis_type: str = "diagnosis", 
                           max_length: int = 256) -> Dict[str, Union[str, float, Dict]]:
        """
        Perform medical analysis on input text using MedGemma
        
        Args:
            text: Medical text to analyze
            analysis_type: Type of analysis ('diagnosis', 'symptom_analysis', 'drug_interaction', etc.)
            max_length: Maximum length of generated response
            
        Returns:
            Dictionary containing analysis results, confidence scores, and metadata
        """
        if not self.model_loaded:
            self.load_model()
            
        try:
            # Get appropriate prompt template
            prompt_template = self.medical_prompts.get(analysis_type, self.medical_prompts["diagnosis"])
            prompt = prompt_template.format(text=text)
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Generate response with medical analysis
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            inference_time = time.time() - start_time
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract generated text (remove input prompt)
            generated_text = response[len(prompt):].strip()
            
            # Calculate confidence score based on response characteristics
            confidence_score = self._calculate_confidence(generated_text, analysis_type)
            
            # Extract medical entities and keywords
            medical_entities = self._extract_medical_entities(generated_text)
            
            analysis_result = {
                "analysis_type": analysis_type,
                "input_text": text,
                "generated_analysis": generated_text,
                "confidence_score": confidence_score,
                "medical_entities": medical_entities,
                "inference_time_seconds": inference_time,
                "timestamp": datetime.now().isoformat(),
                "model_info": {
                    "model_name": "MedGemma-2B",
                    "device": "Raspberry Pi 5 (CPU)",
                    "precision": "float16"
                }
            }
            
            # Memory cleanup
            del outputs, inputs
            gc.collect()
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Error during MedGemma analysis: {e}")
            return {
                "analysis_type": analysis_type,
                "input_text": text,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def batch_analyze(self, texts: List[str], analysis_type: str = "diagnosis", 
                     batch_size: int = 4) -> List[Dict]:
        """
        Perform batch medical analysis for multiple texts
        
        Args:
            texts: List of medical texts to analyze
            analysis_type: Type of analysis to perform
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of analysis results
        """
        if not self.model_loaded:
            self.load_model()
            
        results = []
        
        # Process in batches to manage memory on Raspberry Pi
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
            
            batch_results = []
            for text in batch_texts:
                result = self.analyze_medical_text(text, analysis_type)
                batch_results.append(result)
                
                # Small delay to prevent overheating on Raspberry Pi
                time.sleep(0.1)
            
            results.extend(batch_results)
            
            # Force garbage collection between batches
            gc.collect()
            
        return results
    
    def _calculate_confidence(self, generated_text: str, analysis_type: str) -> float:
        """
        Calculate confidence score based on response characteristics
        
        Args:
            generated_text: Generated medical analysis text
            analysis_type: Type of analysis performed
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            confidence = 0.5  # Base confidence
            
            # Length-based confidence (reasonable length indicates thorough analysis)
            if 50 <= len(generated_text) <= 500:
                confidence += 0.2
            
            # Medical terminology presence
            medical_terms = [
                "diagnosis", "symptoms", "treatment", "patient", "clinical",
                "medical", "condition", "disease", "therapy", "medication",
                "examination", "findings", "assessment", "prognosis"
            ]
            
            term_count = sum(1 for term in medical_terms if term.lower() in generated_text.lower())
            confidence += min(term_count * 0.03, 0.2)
            
            # Structure indicators (lists, numbered points)
            if any(marker in generated_text for marker in ["1.", "2.", "3.", "-", "•"]):
                confidence += 0.1
            
            # Specific analysis type indicators
            if analysis_type == "diagnosis" and any(word in generated_text.lower() 
                                                 for word in ["differential", "likely", "possible"]):
                confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception:
            return 0.5
    
    def _extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities from generated text using simple keyword matching
        
        Args:
            text: Medical analysis text
            
        Returns:
            Dictionary of extracted medical entities by category
        """
        entities = {
            "conditions": [],
            "medications": [],
            "procedures": [],
            "symptoms": [],
            "body_parts": []
        }
        
        # Simple keyword-based extraction (can be enhanced with NER models)
        condition_keywords = ["pneumonia", "diabetes", "hypertension", "infection", "cancer", 
                            "asthma", "arthritis", "depression", "anxiety", "fever"]
        medication_keywords = ["antibiotic", "insulin", "aspirin", "ibuprofen", "acetaminophen",
                             "medication", "drug", "prescription", "treatment"]
        procedure_keywords = ["surgery", "biopsy", "endoscopy", "mri", "ct scan", "x-ray",
                            "ultrasound", "ecg", "blood test"]
        symptom_keywords = ["pain", "fatigue", "nausea", "dizziness", "headache", "cough",
                          "shortness of breath", "chest pain", "abdominal pain"]
        body_part_keywords = ["heart", "lung", "liver", "kidney", "brain", "stomach",
                            "chest", "abdomen", "head", "back"]
        
        text_lower = text.lower()
        
        for keyword in condition_keywords:
            if keyword in text_lower:
                entities["conditions"].append(keyword)
        
        for keyword in medication_keywords:
            if keyword in text_lower:
                entities["medications"].append(keyword)
                
        for keyword in procedure_keywords:
            if keyword in text_lower:
                entities["procedures"].append(keyword)
                
        for keyword in symptom_keywords:
            if keyword in text_lower:
                entities["symptoms"].append(keyword)
                
        for keyword in body_part_keywords:
            if keyword in text_lower:
                entities["body_parts"].append(keyword)
        
        return entities
    
    def generate_clinical_summary(self, patient_data: Dict) -> Dict[str, str]:
        """
        Generate comprehensive clinical summary from structured patient data
        
        Args:
            patient_data: Dictionary containing patient information
            
        Returns:
            Dictionary with clinical summary components
        """
        if not self.model_loaded:
            self.load_model()
            
        try:
            # Format patient data into narrative text
            patient_text = self._format_patient_data(patient_data)
            
            # Generate clinical summary
            summary_result = self.analyze_medical_text(
                patient_text, 
                analysis_type="clinical_summary",
                max_length=400
            )
            
            return {
                "clinical_summary": summary_result["generated_analysis"],
                "confidence": summary_result["confidence_score"],
                "processing_time": summary_result["inference_time_seconds"],
                "timestamp": summary_result["timestamp"]
            }
            
        except Exception as e:
            logger.error(f"Error generating clinical summary: {e}")
            return {"error": str(e)}
    
    def _format_patient_data(self, patient_data: Dict) -> str:
        """Format structured patient data into narrative text"""
        formatted_parts = []
        
        if "chief_complaint" in patient_data:
            formatted_parts.append(f"Chief Complaint: {patient_data['chief_complaint']}")
        
        if "history" in patient_data:
            formatted_parts.append(f"History: {patient_data['history']}")
            
        if "physical_exam" in patient_data:
            formatted_parts.append(f"Physical Examination: {patient_data['physical_exam']}")
            
        if "lab_results" in patient_data:
            formatted_parts.append(f"Laboratory Results: {patient_data['lab_results']}")
            
        if "medications" in patient_data:
            formatted_parts.append(f"Current Medications: {patient_data['medications']}")
        
        return "\n".join(formatted_parts)

def perform_medgemma_inference(data, data_type, analysis_type="diagnosis", batch_size=4):
    """
    Main inference function for MedGemma analysis
    
    Args:
        data: Input data (text, generator, or list)
        data_type: Type of medical data ('mimic', 'mt', 'clinical_notes', etc.)
        analysis_type: Type of medical analysis to perform
        batch_size: Batch size for processing
        
    Returns:
        Analysis results with medical insights and metadata
    """
    analyzer = MedGemmaAnalyzer()
    
    try:
        logger.info(f"Starting MedGemma inference for {data_type} data")
        
        if isinstance(data, Generator):
            # Handle generator input (for datasets like MIMIC)
            all_results = []
            
            for batch_data in data:
                if isinstance(batch_data, tuple):
                    texts, sensitive_features = batch_data
                    
                    if isinstance(texts, np.ndarray):
                        texts = texts.tolist()
                    
                    # Perform batch analysis
                    batch_results = analyzer.batch_analyze(
                        texts, 
                        analysis_type=analysis_type,
                        batch_size=batch_size
                    )
                    
                    # Add sensitive features to results
                    for i, result in enumerate(batch_results):
                        if i < len(sensitive_features):
                            result["sensitive_features"] = sensitive_features.iloc[i].to_dict()
                    
                    all_results.extend(batch_results)
            
            return {
                'predictions': all_results,
                'analysis_type': analysis_type,
                'total_samples': len(all_results),
                'model_type': 'MedGemma'
            }
            
        elif isinstance(data, (list, np.ndarray, pd.Series)):
            # Handle direct list/array input
            if isinstance(data, (np.ndarray, pd.Series)):
                data = data.tolist()
            
            results = analyzer.batch_analyze(
                data,
                analysis_type=analysis_type,
                batch_size=batch_size
            )
            
            return {
                'predictions': results,
                'analysis_type': analysis_type,
                'total_samples': len(results),
                'model_type': 'MedGemma'
            }
            
        elif isinstance(data, str):
            # Handle single text input
            result = analyzer.analyze_medical_text(data, analysis_type=analysis_type)
            
            return {
                'predictions': [result],
                'analysis_type': analysis_type,
                'total_samples': 1,
                'model_type': 'MedGemma'
            }
            
        else:
            raise ValueError(f"Unsupported data type for MedGemma inference: {type(data)}")
            
    except Exception as e:
        logger.error(f"Error in MedGemma inference: {e}")
        return {
            'error': str(e),
            'analysis_type': analysis_type,
            'model_type': 'MedGemma',
            'timestamp': datetime.now().isoformat()
        }

if __name__ == "__main__":
    # Test MedGemma analyzer
    analyzer = MedGemmaAnalyzer()
    
    # Test with sample medical text
    test_text = """
    Patient presents with chest pain, shortness of breath, and fatigue for the past 3 days.
    Blood pressure 140/90 mmHg, heart rate 95 bpm. ECG shows ST elevation in leads II, III, aVF.
    Troponin I elevated at 2.5 ng/mL.
    """
    
    result = analyzer.analyze_medical_text(test_text, analysis_type="diagnosis")
    print(json.dumps(result, indent=2))