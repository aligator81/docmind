"""
Document Processing Service

This service replaces the functionality of 1-extraction.py by providing
document extraction capabilities within the FastAPI backend.

Features:
- Extract text from PDF, DOCX, PPTX, XLSX, HTML, MD, and image files
- OCR support for scanned documents
- Performance optimizations and caching
- Batch processing capabilities
- Fallback logic for different file types
"""

import os
import sys
import warnings
import base64
import tempfile
import requests
import json
import hashlib
import time
import psutil
from datetime import datetime
from typing import Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import asyncio

# Document processing imports
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from mistralai import Mistral
from PIL import Image
from io import BytesIO
import fitz

# GPU acceleration imports (optional)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

# PDF processing imports
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False
    PyPDF2 = None

# Fix Unicode encoding issues on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
warnings.filterwarnings("ignore", message=".*clean_up_tokenization_spaces.*")

@dataclass
class ProcessingResult:
    """Document processing result data class"""
    success: bool
    content: str
    method: str
    processing_time: float
    metadata: Dict = None

@dataclass
class PerformanceMetrics:
    """Performance metrics data class"""
    total_time: float
    pages_per_second: float
    memory_usage_mb: float
    cpu_usage_percent: float
    timestamp: str
    processed_pages: int
    extraction_method: str

class PerformanceTracker:
    """Track and log performance metrics"""

    def __init__(self, log_file: str = "performance_log.txt"):
        self.log_file = log_file
        self.metrics_history = []

    def track_performance(self, func: Callable):
        """Decorator to track function performance"""
        def wrapper(*args, **kwargs):
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss
            start_cpu = process.cpu_percent()

            result = func(*args, **kwargs)

            end_time = time.time()
            end_memory = process.memory_info().rss
            end_cpu = process.cpu_percent()

            page_count = kwargs.get('page_count', 1)
            method = kwargs.get('method', 'unknown')

            metrics = PerformanceMetrics(
                total_time=end_time - start_time,
                pages_per_second=page_count / max(0.001, end_time - start_time),
                memory_usage_mb=(end_memory - start_memory) / 1024 / 1024,
                cpu_usage_percent=end_cpu - start_cpu,
                timestamp=datetime.now().isoformat(),
                processed_pages=page_count,
                extraction_method=method
            )

            self.metrics_history.append(metrics)
            self._log_metrics(metrics, func.__name__)
            return result

        return wrapper

    def _log_metrics(self, metrics: PerformanceMetrics, function_name: str):
        """Log performance metrics"""
        log_entry = (
            f"{metrics.timestamp} | {function_name} | "
            f"Method: {metrics.extraction_method} | "
            f"Time: {metrics.total_time:.2f}s | "
            f"Pages/s: {metrics.pages_per_second:.2f} | "
            f"Memory: {metrics.memory_usage_mb:.2f}MB | "
            f"CPU: {metrics.cpu_usage_percent:.1f}% | "
            f"Pages: {metrics.processed_pages}"
        )

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_entry + "\n")

        print(log_entry)

class DocumentCache:
    """File-based caching system for document processing results"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def get_file_hash(self, file_path: str) -> str:
        """Generate unique hash for file content"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def get_cached_result(self, file_path: str, operation: str) -> Optional[dict]:
        """Get cached result if exists"""
        cache_key = f"{self.get_file_hash(file_path)}_{operation}.json"
        cache_file = self.cache_dir / cache_key

        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return None
        return None

    def cache_result(self, file_path: str, operation: str, result: dict):
        """Cache processing result"""
        cache_key = f"{self.get_file_hash(file_path)}_{operation}.json"
        cache_file = self.cache_dir / cache_key

        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

class DocumentProcessor:
    """Unified document processing service"""

    def __init__(self):
        self.performance_tracker = PerformanceTracker()
        self.document_cache = DocumentCache()
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.gpu_available = self._check_gpu_availability()

    def _check_gpu_availability(self, force_gpu: bool = False, force_cpu: bool = False) -> bool:
        """Check if GPU acceleration is available"""
        if not TORCH_AVAILABLE:
            print("🔶 GPU acceleration not available: PyTorch not installed")
            return False

        if force_cpu:
            print("🖥️ CPU mode forced by user")
            return False

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown GPU"

            if gpu_count > 0 and hasattr(torch.cuda, 'get_device_properties'):
                gpu_props = torch.cuda.get_device_properties(0)
                gpu_memory_gb = gpu_props.total_memory / (1024**3)
                print(f"✅ GPU acceleration available: {gpu_name} ({gpu_memory_gb:.1f}GB)")
            else:
                print(f"✅ GPU acceleration available: {gpu_name} ({gpu_count} GPU(s))")

            return True
        else:
            if force_gpu:
                print("🔶 Forced GPU mode requested but no CUDA-compatible GPU found")
            else:
                print("🔶 GPU acceleration not available: No CUDA-compatible GPU found")
            return False

    def _get_accelerator_config(self, force_gpu: bool = False, force_cpu: bool = False, gpu_memory_limit: float = None) -> Dict:
        """Get accelerator configuration for Docling"""
        config = {}

        # Check GPU availability with user preferences
        gpu_available = self._check_gpu_availability(force_gpu, force_cpu)

        if gpu_available and torch.cuda.is_available():
            # Use GPU acceleration - simplified configuration to avoid dict attribute error
            config["accelerator"] = "cuda"
            print("🚀 Using GPU acceleration for Docling")
        else:
            # Use CPU
            config["accelerator"] = "cpu"
            print("🖥️ Using CPU for Docling")

        return config

    def _get_supported_formats(self) -> Dict[str, InputFormat]:
        """Get supported file formats mapping"""
        return {
            '.pdf': InputFormat.PDF,
            '.docx': InputFormat.DOCX,
            '.pptx': InputFormat.PPTX,
            '.xlsx': InputFormat.XLSX,
            '.html': InputFormat.HTML,
            '.htm': InputFormat.HTML,
            '.md': InputFormat.MD,
            '.png': InputFormat.IMAGE,
            '.jpg': InputFormat.IMAGE,
            '.jpeg': InputFormat.IMAGE,
            '.tiff': InputFormat.IMAGE,
            '.bmp': InputFormat.IMAGE,
        }

    def _encode_file_to_base64(self, file_path: str) -> Optional[str]:
        """Encode file to base64 string"""
        try:
            with open(file_path, "rb") as file:
                return base64.b64encode(file.read()).decode('utf-8')
        except Exception as e:
            print(f"❌ Error encoding file: {e}")
            return None

    def _save_to_file(self, content: str, filename: str) -> bool:
        """Save content to file"""
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            return True
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return False

    def _get_mistral_client(self) -> Optional[Mistral]:
        """Get Mistral client"""
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            print("❌ MISTRAL_API_KEY not found in environment variables")
            return None
        return Mistral(api_key=api_key)

    async def extract_with_docling(self, file_path: str, enable_ocr: bool = False, original_filename: Optional[str] = None) -> ProcessingResult:
        """Extract document using Docling (local processing)"""
        try:
            # Check cache first
            cached_result = self.document_cache.get_cached_result(file_path, "docling_extraction")
            if cached_result:
                print(f"📋 Using cached Docling result for {file_path}")
                # Update filename to use original if available
                output_name = os.path.splitext(original_filename or os.path.basename(file_path))[0]
                return ProcessingResult(
                    success=True,
                    content=cached_result.get('content', ''),
                    method='cached',
                    processing_time=0.0,
                    metadata=cached_result
                )

            output_name = os.path.splitext(original_filename or os.path.basename(file_path))[0]
            filename = f"{self.output_dir}/{output_name}_docling_extracted.md"

            # Check file size
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            print(f"🔍 Extracting with Docling: {file_path} ({file_size_mb:.1f}MB)")

            if file_size_mb > 50:  # Warn for very large files
                print(f"⚠️ Warning: Large file detected ({file_size_mb:.1f}MB). This may take a while...")

            # Get file format
            file_extension = os.path.splitext(file_path)[1].lower()

            # Special handling for markdown files - use simple text reading to avoid Docling issues
            if file_extension == '.md':
                print("📝 Using simple markdown reader for .md files")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Save to file
                    if self._save_to_file(content, filename):
                        print(f"✅ Markdown extraction completed! Content saved to: {filename}")
                        print(f"📄 Extracted {len(content)} characters")

                        # Cache the result
                        self.document_cache.cache_result(file_path, "docling_extraction", {
                            "content": content,
                            "filename": filename,
                            "file_extension": file_extension,
                            "enable_ocr": enable_ocr,
                            "processing_time": 0.1,
                            "original_filename": original_filename
                        })

                        return ProcessingResult(
                            success=True,
                            content=content,
                            method="markdown_simple",
                            processing_time=0.1,
                            metadata={
                                "filename": filename,
                                "file_extension": file_extension,
                                "enable_ocr": enable_ocr,
                                "original_filename": original_filename
                            }
                        )
                except Exception as md_error:
                    print(f"❌ Simple markdown reading failed: {md_error}")
                    return ProcessingResult(
                        success=False,
                        content="",
                        method="markdown_read_error",
                        processing_time=0.0
                    )

            format_mapping = self._get_supported_formats()

            if file_extension not in format_mapping:
                print(f"❌ Unsupported file format for Docling: {file_extension}")
                return ProcessingResult(
                    success=False,
                    content="",
                    method="unsupported_format",
                    processing_time=0.0
                )

            input_format = format_mapping[file_extension]

            # Create converter with minimal configuration
            print("🔄 Initializing Docling converter...")
            try:
                converter = DocumentConverter()
            except Exception as config_error:
                print(f"❌ DocumentConverter initialization failed: {config_error}")
                return ProcessingResult(
                    success=False,
                    content="",
                    method="converter_init_error",
                    processing_time=0.0
                )

            print("🔄 Starting document conversion (this may take several minutes for large files)...")
            start_time = time.time()

            # Adjust timeout based on file type - images need more time for OCR
            if file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                timeout_seconds = max(120, min(int(file_size_mb * 5), 3600))  # Max 60 minutes for images
            else:
                timeout_seconds = max(60, min(int(file_size_mb * 2), 1800))  # Max 30 minutes for other files
            
            print(f"⏱️ Setting timeout to {timeout_seconds} seconds ({timeout_seconds/60:.1f} minutes)")

            try:
                # Use asyncio for timeout handling
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, converter.convert, file_path),
                    timeout=timeout_seconds
                )

                if not result:
                    print("❌ No result from document conversion")
                    return ProcessingResult(
                        success=False,
                        content="",
                        method="conversion_error",
                        processing_time=time.time() - start_time
                    )

                doc = result.document
                processing_time = time.time() - start_time
                print(f"✅ Document conversion completed in {processing_time:.1f} seconds")

                # Export to markdown with enhanced metadata preservation
                print("📝 Exporting to markdown...")
                content = doc.export_to_markdown()

                # Save to file
                if self._save_to_file(content, filename):
                    print(f"✅ Docling extraction completed! Content saved to: {filename}")
                    print(f"📄 Extracted {len(content)} characters in {processing_time:.1f} seconds")

                    # Cache the result
                    self.document_cache.cache_result(file_path, "docling_extraction", {
                        "content": content,
                        "filename": filename,
                        "file_extension": file_extension,
                        "enable_ocr": enable_ocr,
                        "processing_time": processing_time,
                        "original_filename": original_filename
                    })

                    return ProcessingResult(
                        success=True,
                        content=content,
                        method="docling",
                        processing_time=processing_time,
                        metadata={
                            "filename": filename,
                            "file_extension": file_extension,
                            "enable_ocr": enable_ocr,
                            "original_filename": original_filename
                        }
                    )
                else:
                    return ProcessingResult(
                        success=False,
                        content="",
                        method="save_error",
                        processing_time=processing_time
                    )

            except asyncio.TimeoutError:
                print(f"⏰ Document conversion timed out after {timeout_seconds} seconds")
                print("💡 Tip: For complex images, try using Mistral OCR or simplify the image")
                return ProcessingResult(
                    success=False,
                    content="",
                    method="timeout",
                    processing_time=timeout_seconds
                )

        except Exception as e:
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            print(f"❌ Docling extraction failed: {e}")
            return ProcessingResult(
                success=False,
                content="",
                method="error",
                processing_time=processing_time
            )

    async def extract_with_mistral_ocr(self, file_path: str, original_filename: Optional[str] = None) -> ProcessingResult:
        """Extract document using Mistral OCR (cloud processing)"""
        try:
            output_name = os.path.splitext(original_filename or os.path.basename(file_path))[0]
            filename = f"{self.output_dir}/{output_name}_mistral_extracted.md"

            print(f"☁️ Extracting with Mistral OCR: {file_path}")

            # Get Mistral client
            mistral_client = self._get_mistral_client()
            if not mistral_client:
                return ProcessingResult(
                    success=False,
                    content="",
                    method="no_api_key",
                    processing_time=0.0
                )

            # Determine file type and prepare document data
            file_extension = os.path.splitext(file_path)[1].lower()

            if file_extension in ['.png', '.jpg', '.jpeg', '.webp', '.gif']:
                # Image file
                base64_data = self._encode_file_to_base64(file_path)
                if not base64_data:
                    return ProcessingResult(
                        success=False,
                        content="",
                        method="encoding_error",
                        processing_time=0.0
                    )

                document_data = {
                    "type": "image_url",
                    "image_url": f"data:image/{file_extension[1:]};base64,{base64_data}"
                }

            elif file_extension in ['.pdf', '.pptx', '.docx']:
                # Document file
                base64_data = self._encode_file_to_base64(file_path)
                if not base64_data:
                    return ProcessingResult(
                        success=False,
                        content="",
                        method="encoding_error",
                        processing_time=0.0
                    )

                mime_type = {
                    '.pdf': 'application/pdf',
                    '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
                    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document'
                }[file_extension]

                document_data = {
                    "type": "document_url",
                    "document_url": f"data:{mime_type};base64,{base64_data}"
                }

            else:
                print(f"❌ Unsupported file format for Mistral OCR: {file_extension}")
                return ProcessingResult(
                    success=False,
                    content="",
                    method="unsupported_format",
                    processing_time=0.0
                )

            # Process with Mistral OCR
            start_time = time.time()
            ocr_response = mistral_client.ocr.process(
                model="mistral-ocr-latest",
                document=document_data,
                include_image_base64=False
            )

            processing_time = time.time() - start_time

            # Extract text content
            if hasattr(ocr_response, 'pages') and ocr_response.pages:
                content = ""
                for page in ocr_response.pages:
                    if hasattr(page, 'markdown') and page.markdown:
                        content += page.markdown + "\n\n"
                    elif hasattr(page, 'content') and page.content:
                        content += page.content + "\n\n"
                else:
                    print("❌ No content extracted from document with Mistral OCR")
                    return ProcessingResult(
                        success=False,
                        content="",
                        method="no_content",
                        processing_time=processing_time
                    )

            # Save to file
            if self._save_to_file(content, filename):
                print(f"✅ Mistral OCR extraction completed! Content saved to: {filename}")
                print(f"📄 Extracted {len(content)} characters")

                return ProcessingResult(
                    success=True,
                    content=content,
                    method="mistral_ocr",
                    processing_time=processing_time,
                    metadata={
                        "filename": filename,
                        "file_extension": file_extension,
                        "original_filename": original_filename
                    }
                )
            else:
                return ProcessingResult(
                    success=False,
                    content="",
                    method="save_error",
                    processing_time=processing_time
                )

        except Exception as e:
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            print(f"❌ Mistral OCR extraction failed: {e}")
            return ProcessingResult(
                success=False,
                content="",
                method="error",
                processing_time=processing_time
            )

    async def extract_document(self, file_path: str, prefer_cloud: bool = False, use_cache: bool = True, original_filename: Optional[str] = None) -> ProcessingResult:
        """
        Extract document with intelligent fallback logic

        Args:
            file_path: Path to document
            prefer_cloud: Prefer Mistral OCR over Docling
            use_cache: Use caching for results

        Returns:
            ProcessingResult with extraction details
        """
        print(f"🚀 Processing document: {file_path}")

        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return ProcessingResult(
                success=False,
                content="",
                method="file_not_found",
                processing_time=0.0
            )

        # Check cache if enabled
        if use_cache:
            cached_result = self.document_cache.get_cached_result(file_path, "unified_extraction")
            if cached_result:
                print(f"📋 Using cached result for {file_path}")
                # Update filename to use original if available
                if original_filename:
                    output_name = os.path.splitext(original_filename)[0]
                    method = cached_result.get('method', 'unknown')
                    if method == 'docling':
                        cached_result['filename'] = f"{self.output_dir}/{output_name}_docling_extracted.md"
                    elif method == 'mistral_ocr':
                        cached_result['filename'] = f"{self.output_dir}/{output_name}_mistral_extracted.md"
                return ProcessingResult(
                    success=True,
                    content=cached_result.get('content', ''),
                    method='cached',
                    processing_time=0.0,
                    metadata=cached_result
                )

        # Get file extension for special handling
        file_extension = os.path.splitext(file_path)[1].lower()

        # Special handling for markdown files - avoid Docling issues
        if file_extension == '.md':
            print("📝 Using simple extraction for markdown files")
            result = self._extract_markdown_simple(file_path)
            if result.success:
                self.document_cache.cache_result(file_path, "unified_extraction", {
                    "content": result.content,
                    "method": result.method,
                    "processing_time": result.processing_time,
                    "original_filename": original_filename
                })
            return result

        # Try preferred method first for non-markdown files
        if prefer_cloud:
            print("☁️ Trying Mistral OCR first...")
            result = await self.extract_with_mistral_ocr(file_path, original_filename)
            if result.success:
                self.document_cache.cache_result(file_path, "unified_extraction", {
                    "content": result.content,
                    "method": result.method,
                    "processing_time": result.processing_time,
                    "original_filename": original_filename
                })
                return result

            print("⚠️ Mistral OCR failed, trying Docling...")
            result = await self.extract_with_docling(file_path, original_filename=original_filename)
        else:
            print("🔍 Trying Docling first...")
            result = await self.extract_with_docling(file_path)
            if result.success:
                self.document_cache.cache_result(file_path, "unified_extraction", {
                    "content": result.content,
                    "method": result.method,
                    "processing_time": result.processing_time,
                    "original_filename": original_filename
                })
                return result

            print("⚠️ Docling failed, trying Mistral OCR...")
            result = await self.extract_with_mistral_ocr(file_path)

        # If both Docling and Mistral OCR fail, try direct EasyOCR for images
        if not result.success and file_extension in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
            print("⚠️ Both Docling and Mistral OCR failed, trying direct EasyOCR...")
            result = await self._extract_with_easyocr(file_path)

        # If both Docling and Mistral OCR fail, try PyPDF2 for PDF files
        if not result.success and file_extension == '.pdf' and PYPDF2_AVAILABLE:
            print("⚠️ Mistral OCR failed, trying PyPDF2 fallback...")
            result = self._extract_with_pypdf2(file_path)

        if result.success:
            self.document_cache.cache_result(file_path, "unified_extraction", {
                "content": result.content,
                "method": result.method,
                "processing_time": result.processing_time
            })

        return result

    def _extract_markdown_simple(self, file_path: str) -> ProcessingResult:
        """Simple markdown file extraction to avoid Docling issues"""
        try:
            start_time = time.time()

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            processing_time = time.time() - start_time

            print(f"✅ Markdown extraction completed in {processing_time:.2f} seconds")
            print(f"📄 Extracted {len(content)} characters")

            return ProcessingResult(
                success=True,
                content=content,
                method="markdown_simple",
                processing_time=processing_time
            )

        except Exception as e:
            print(f"❌ Simple markdown extraction failed: {e}")
            return ProcessingResult(
                success=False,
                content="",
                method="markdown_error",
                processing_time=0.0
            )

    async def _extract_with_easyocr(self, file_path: str) -> ProcessingResult:
        """Extract text from image using direct EasyOCR as fallback"""
        try:
            import easyocr
            
            print(f"🔍 Trying direct EasyOCR extraction for: {file_path}")
            start_time = time.time()
            
            # Initialize EasyOCR reader
            reader = easyocr.Reader(['en'])
            
            # Read the image and extract text
            result = reader.readtext(file_path)
            
            processing_time = time.time() - start_time
            
            if result:
                content = ""
                for detection in result:
                    text = detection[1]
                    confidence = detection[2]
                    if confidence > 0.5:  # Only include high-confidence detections
                        content += text + "\n"
                
                if content.strip():
                    print(f"✅ EasyOCR extraction completed in {processing_time:.2f} seconds")
                    print(f"📄 Extracted {len(content)} characters")
                    return ProcessingResult(
                        success=True,
                        content=content,
                        method="easyocr_direct",
                        processing_time=processing_time,
                        metadata={"detections": len(result)}
                    )
                else:
                    print(f"❌ EasyOCR extracted no high-confidence content from {file_path}")
                    return ProcessingResult(
                        success=False,
                        content="",
                        method="easyocr_no_content",
                        processing_time=processing_time
                    )
            else:
                print(f"❌ EasyOCR found no text in {file_path}")
                return ProcessingResult(
                    success=False,
                    content="",
                    method="easyocr_no_detections",
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            print(f"❌ EasyOCR extraction failed: {e}")
            return ProcessingResult(
                success=False,
                content="",
                method="easyocr_error",
                processing_time=processing_time
            )

    def _extract_with_pypdf2(self, file_path: str) -> ProcessingResult:
        """Extract text from PDF using PyPDF2 as fallback"""
        try:
            if not PYPDF2_AVAILABLE:
                return ProcessingResult(
                    success=False,
                    content="",
                    method="pypdf2_not_available",
                    processing_time=0.0
                )

            start_time = time.time()
            print(f"📄 Trying PyPDF2 extraction for: {file_path}")

            content = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            content += f"--- Page {page_num + 1} ---\n{page_text}\n\n"
                    except Exception as page_error:
                        print(f"⚠️ Error extracting page {page_num + 1}: {page_error}")
                        continue

            processing_time = time.time() - start_time

            if content.strip():
                print(f"✅ PyPDF2 extraction completed in {processing_time:.2f} seconds")
                print(f"📄 Extracted {len(content)} characters from {len(pdf_reader.pages)} pages")
                return ProcessingResult(
                    success=True,
                    content=content,
                    method="pypdf2",
                    processing_time=processing_time,
                    metadata={"pages_processed": len(pdf_reader.pages)}
                )
            else:
                print(f"❌ PyPDF2 extracted no content from {file_path}")
                return ProcessingResult(
                    success=False,
                    content="",
                    method="pypdf2_no_content",
                    processing_time=processing_time
                )

        except Exception as e:
            processing_time = time.time() - start_time if 'start_time' in locals() else 0
            print(f"❌ PyPDF2 extraction failed: {e}")
            return ProcessingResult(
                success=False,
                content="",
                method="pypdf2_error",
                processing_time=processing_time
            )

    async def process_from_database(self, db, mark_processed: bool = False, timeout_hours: float = None) -> int:
        """Process all unprocessed documents from database"""
        from ..models import Document

        try:
            # Get unprocessed documents
            documents = db.query(Document).filter(
                Document.content.is_(None),
                Document.status == "not processed"
            ).all()

            if not documents:
                print("✅ No unprocessed documents found in database")
                return 0

            print(f"📋 Found {len(documents)} unprocessed document(s) in database")

            success_count = 0
            start_time = time.time()

            for doc in documents:
                # Check timeout if specified
                if timeout_hours and (time.time() - start_time) > (timeout_hours * 3600):
                    print(f"\n⏰ Timeout reached after {timeout_hours} hours. Processed {success_count}/{len(documents)} documents.")
                    break

                file_path = doc.file_path
                if file_path and os.path.exists(file_path):
                    print(f"\n🔄 Processing [{success_count + 1}/{len(documents)}]: {doc.filename} (ID: {doc.id})")

                    try:
                        result = await self.extract_document(doc.file_path, original_filename=doc.original_filename)
                    except Exception as e:
                        print(f"❌ Error during extract_document: {e}")
                        result = ProcessingResult(success=False, content="", method="extract_error", processing_time=0.0)

                    if result.success:
                        # Update database with extracted content
                        doc.content = result.content
                        doc.status = "extracted"
                        doc.processed_at = datetime.utcnow()
                        db.commit()

                        print(f"✅ Successfully processed and stored: {doc.filename} using {result.method}")
                        success_count += 1
                    else:
                        print(f"❌ Failed to extract content: {doc.filename}")
                        if result.method == "timeout":
                            print(f"💡 Tip: Large files may need more time. Consider increasing timeout.")
                else:
                    print(f"⚠️ File not found for document {doc.filename}: {file_path}")

            elapsed_total = time.time() - start_time
            print(f"\n🎉 Database processing completed! Successfully processed {success_count}/{len(documents)} documents in {elapsed_total:.1f} seconds")

            return success_count

        except Exception as e:
            print(f"❌ Error in database processing: {e}")
            db.rollback()
            return 0

    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        try:
            with open(self.performance_tracker.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                return {"error": "No performance data available"}

            # Parse performance data
            total_extractions = len(lines)
            methods = {}
            total_time = 0
            total_memory = 0

            for line in lines[-10:]:  # Last 10 entries
                parts = line.strip().split(' | ')
                if len(parts) >= 6:
                    method = parts[2].replace("Method: ", "")
                    time_str = parts[3].replace("Time: ", "").replace("s", "")
                    memory_str = parts[4].replace("Memory: ", "").replace("MB", "")

                    try:
                        total_time += float(time_str)
                        total_memory += float(memory_str)

                        if method not in methods:
                            methods[method] = 0
                        methods[method] += 1
                    except ValueError:
                        continue

            return {
                "total_extractions": total_extractions,
                "recent_extractions": len(lines[-10:]),
                "methods_used": methods,
                "average_time": total_time / max(1, len(lines[-10:])),
                "average_memory": total_memory / max(1, len(lines[-10:])),
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            return {"error": str(e)}