#!/usr/bin/env python3
"""
Test script for DocumentStructureAnalyzer service
"""

import sys
import os
import json

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.document_structure_analyzer import (
    DocumentStructureAnalyzer, 
    DocumentType,
    create_document_structure_analyzer
)

def test_document_type_classification():
    """Test document type classification"""
    print("🧪 Testing document type classification...")
    
    analyzer = create_document_structure_analyzer()
    
    # Test academic paper content
    academic_content = """
    ABSTRACT
    This paper presents a novel approach to document analysis.
    
    INTRODUCTION
    Document structure analysis is crucial for information retrieval.
    
    METHODOLOGY
    We employed machine learning techniques for section detection.
    
    RESULTS
    Our method achieved 95% accuracy in section classification.
    
    DISCUSSION
    The results demonstrate the effectiveness of our approach.
    
    REFERENCES
    1. Smith, J. (2023). Document Analysis. Journal of Computing.
    """
    
    structure = analyzer.analyze_document_structure(academic_content, "research_paper.pdf")
    assert structure.document_type == DocumentType.ACADEMIC_PAPER
    print(f"✅ Academic paper classified as: {structure.document_type}")
    
    # Test legal document content
    legal_content = """
    AGREEMENT
    This Agreement is made between Party A and Party B.
    
    SECTION 1: DEFINITIONS
    For the purposes of this Agreement, the following terms shall have the meanings set forth below.
    
    SECTION 2: OBLIGATIONS
    Party A hereby agrees to provide the services described herein.
    
    CLAUSE 2.1: Payment Terms
    Payment shall be made within 30 days of invoice receipt.
    """
    
    structure = analyzer.analyze_document_structure(legal_content, "contract.pdf")
    assert structure.document_type == DocumentType.LEGAL_DOCUMENT
    print(f"✅ Legal document classified as: {structure.document_type}")
    
    # Test technical manual content
    technical_content = """
    USER MANUAL
    
    CHAPTER 1: INSTALLATION
    Follow these steps to install the software.
    
    PROCEDURE 1.1: System Requirements
    Ensure your system meets the following requirements.
    
    STEP 1: Download the installer
    Visit our website and download the latest version.
    
    WARNING: Do not interrupt the installation process.
    
    TABLE 1: System Specifications
    Minimum requirements for optimal performance.
    """
    
    structure = analyzer.analyze_document_structure(technical_content, "user_manual.pdf")
    assert structure.document_type == DocumentType.TECHNICAL_MANUAL
    print(f"✅ Technical manual classified as: {structure.document_type}")

def test_section_detection():
    """Test hierarchical section detection"""
    print("🧪 Testing hierarchical section detection...")
    
    analyzer = create_document_structure_analyzer()
    
    content = """
    EXECUTIVE SUMMARY
    This report provides an overview of market trends.
    
    MARKET ANALYSIS
    Current market conditions and future projections.
    
    SECTION 1: MARKET SIZE
    The global market size is estimated at $100 billion.
    
    SUBSECTION 1.1: Regional Breakdown
    North America accounts for 40% of the market.
    
    FINANCIAL PROJECTIONS
    Revenue growth is expected to be 15% annually.
    
    RECOMMENDATIONS
    We recommend expanding into emerging markets.
    """
    
    structure = analyzer.analyze_document_structure(content, "business_report.pdf")
    
    # Verify sections were detected
    assert len(structure.sections) > 0
    print(f"✅ Detected {len(structure.sections)} root sections")
    
    # Verify section hierarchy
    for section in structure.sections:
        print(f"  - Level {section.level}: {section.title} (confidence: {section.confidence:.2f})")
        for subsection in section.subsections:
            print(f"    - Level {subsection.level}: {subsection.title} (confidence: {subsection.confidence:.2f})")
    
    # Verify confidence scores
    assert 'overall' in structure.confidence_scores
    assert structure.confidence_scores['overall'] > 0
    print(f"✅ Overall confidence: {structure.confidence_scores['overall']:.2f}")

def test_page_marker_extraction():
    """Test page number extraction"""
    print("🧪 Testing page marker extraction...")
    
    analyzer = create_document_structure_analyzer()
    
    content = """
    INTRODUCTION
    This document spans multiple pages.
    
    Page 2
    This is the content on page 2.
    
    SECTION 1: OVERVIEW
    Continuing with the document content.
    
    p. 3
    Now we're on page 3.
    
    - 4 -
    Page 4 content with centered marker.
    
    SECTION 2: DETAILS
    More detailed information.
    
    page 5
    Final page content.
    """
    
    structure = analyzer.analyze_document_structure(content, "multi_page_doc.pdf")
    
    # Verify page count estimation
    assert structure.page_count >= 4  # Should detect at least 4 pages
    print(f"✅ Estimated page count: {structure.page_count}")
    
    # Verify metadata
    assert 'page_markers_found' in structure.metadata
    print(f"✅ Page markers found: {structure.metadata['page_markers_found']}")

def test_enhanced_metadata_extraction():
    """Test enhanced metadata extraction"""
    print("🧪 Testing enhanced metadata extraction...")
    
    analyzer = create_document_structure_analyzer()
    
    content = """
    RESEARCH PAPER ON MACHINE LEARNING
    
    ABSTRACT
    This paper explores deep learning applications.
    
    1. INTRODUCTION
    Machine learning has revolutionized many fields.
    
    1.1 Background
    Traditional methods have limitations.
    
    2. METHODOLOGY
    We used convolutional neural networks.
    
    2.1 Data Collection
    Dataset consisted of 10,000 images.
    
    3. RESULTS
    Our model achieved 98% accuracy.
    
    Page 2
    4. DISCUSSION
    The results are promising for future applications.
    
    5. CONCLUSION
    Deep learning shows great potential.
    
    REFERENCES
    [1] LeCun, Y. (2015). Deep learning. Nature.
    """
    
    metadata = analyzer.extract_enhanced_metadata(content, "research_paper.pdf")
    
    # Verify metadata structure
    required_fields = [
        'document_type', 'page_count', 'section_count', 
        'max_section_depth', 'confidence_scores', 'sections_hierarchy'
    ]
    
    for field in required_fields:
        assert field in metadata
        print(f"✅ {field}: {metadata[field]}")
    
    # Verify sections hierarchy
    assert isinstance(metadata['sections_hierarchy'], list)
    print(f"✅ Sections hierarchy has {len(metadata['sections_hierarchy'])} root sections")
    
    # Print detailed metadata
    print(f"📊 Enhanced metadata: {json.dumps(metadata, indent=2)}")

def test_confidence_scoring():
    """Test confidence score calculation"""
    print("🧪 Testing confidence scoring...")
    
    analyzer = create_document_structure_analyzer()
    
    # Test with good structure
    good_content = """
    EXECUTIVE SUMMARY
    Clear heading structure.
    
    SECTION 1: OVERVIEW
    Well-defined sections.
    
    Page 2
    Multiple page markers.
    
    SECTION 2: DETAILS
    More sections with proper formatting.
    """
    
    structure = analyzer.analyze_document_structure(good_content, "well_structured.pdf")
    
    # Should have high confidence for well-structured document
    assert structure.confidence_scores['overall'] > 0.6
    print(f"✅ Well-structured document confidence: {structure.confidence_scores['overall']:.2f}")
    
    # Test with poor structure
    poor_content = """
    This is just some random text without clear structure.
    No proper headings or page markers.
    Just continuous paragraphs without organization.
    It's difficult to extract meaningful structure from this.
    """
    
    structure = analyzer.analyze_document_structure(poor_content, "unstructured.txt")
    
    # Should have lower confidence for unstructured document
    assert structure.confidence_scores['overall'] < 0.5
    print(f"✅ Unstructured document confidence: {structure.confidence_scores['overall']:.2f}")

def test_edge_cases():
    """Test edge cases and error handling"""
    print("🧪 Testing edge cases...")
    
    analyzer = create_document_structure_analyzer()
    
    # Test empty content
    structure = analyzer.analyze_document_structure("", "empty.txt")
    assert structure.document_type == DocumentType.UNKNOWN
    assert len(structure.sections) == 0
    print("✅ Empty content handled correctly")
    
    # Test very short content
    short_content = "Short doc"
    structure = analyzer.analyze_document_structure(short_content, "short.txt")
    assert structure.confidence_scores['overall'] < 0.5
    print("✅ Short content handled correctly")
    
    # Test content with only page markers
    page_only_content = """
    Page 1
    Some content here.
    
    Page 2
    More content here.
    
    Page 3
    Final content.
    """
    
    structure = analyzer.analyze_document_structure(page_only_content, "pages_only.pdf")
    assert structure.page_count >= 3
    print("✅ Page-only content handled correctly")

def main():
    """Run all DocumentStructureAnalyzer tests"""
    print("🚀 Starting Document Structure Analyzer Tests")
    print("=" * 50)
    
    try:
        test_document_type_classification()
        test_section_detection()
        test_page_marker_extraction()
        test_enhanced_metadata_extraction()
        test_confidence_scoring()
        test_edge_cases()
        
        print("\n🎉 All DocumentStructureAnalyzer tests completed successfully!")
        print("📊 The analyzer can now:")
        print("   - Classify document types (academic, legal, technical, business)")
        print("   - Detect hierarchical sections with confidence scoring")
        print("   - Extract page numbers and estimate page count")
        print("   - Build section hierarchies with parent-child relationships")
        print("   - Calculate comprehensive confidence scores")
        print("   - Extract enhanced metadata for improved search and citation")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())