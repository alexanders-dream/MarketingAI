"""
Export utilities for Market Intelligence - PDF and Chart exports
"""
import logging
import io
import base64
from typing import Dict, Any, List, Optional
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


class MarketIntelligenceExporter:
    """Export market intelligence data to PDF and charts"""
    
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for PDF generation"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1f77b4')
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.HexColor('#2c3e50')
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomSubHeading',
            parent=self.styles['Heading3'],
            fontSize=13,
            spaceBefore=15,
            spaceAfter=8,
            textColor=colors.HexColor('#34495e')
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leading=16
        ))
    
    def export_to_pdf(self, analysis_data: Dict[str, str], charts: Optional[List[Dict[str, Any]]] = None) -> bytes:
        """
        Export market analysis to PDF
        
        Args:
            analysis_data: Market analysis data dictionary
            charts: Optional list of chart data (plotly figures as dicts)
            
        Returns:
            PDF file as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        story = []
        
        # Title
        title = Paragraph("Market Intelligence Report", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Date
        date_text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        story.append(Paragraph(date_text, self.styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Company Overview
        story.append(Paragraph("Company Overview", self.styles['CustomHeading']))
        company_info = self._format_company_info(analysis_data)
        story.append(Paragraph(company_info, self.styles['CustomBody']))
        story.append(Spacer(1, 12))
        
        # Market Analysis Sections
        sections = [
            ("Competitive Landscape", "competitors"),
            ("Market Trends", "market_trends"),
            ("Market Opportunities", "market_opportunities"),
            ("Target Segments", "target_segments"),
            ("Competitive Advantages", "competitive_advantages"),
            ("Market Size & Growth", "market_size")
        ]
        
        for title, key in sections:
            if analysis_data.get(key) and analysis_data[key] != "Analysis failed":
                story.append(Paragraph(title, self.styles['CustomHeading']))
                content = self._clean_markdown_for_pdf(analysis_data[key])
                story.append(Paragraph(content, self.styles['CustomBody']))
                story.append(Spacer(1, 8))
        
        # Add charts if provided
        if charts:
            story.append(PageBreak())
            story.append(Paragraph("Market Analysis Charts", self.styles['CustomHeading']))
            story.append(Spacer(1, 12))
            
            for chart_data in charts:
                if chart_data.get('image_data'):
                    story.append(Paragraph(chart_data.get('title', 'Chart'), self.styles['CustomSubHeading']))
                    story.append(Spacer(1, 8))
                    
                    # Add chart image
                    img = Image(chart_data['image_data'], width=6*inch, height=4*inch)
                    story.append(img)
                    story.append(Spacer(1, 20))
        
        # Comprehensive Analysis
        if analysis_data.get("comprehensive_analysis") and analysis_data["comprehensive_analysis"] != "Analysis failed":
            story.append(PageBreak())
            story.append(Paragraph("Comprehensive Market Analysis", self.styles['CustomHeading']))
            comprehensive = self._clean_markdown_for_pdf(analysis_data["comprehensive_analysis"])
            story.append(Paragraph(comprehensive, self.styles['CustomBody']))
        
        # Build PDF
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        logger.info(f"PDF export completed successfully ({len(pdf_bytes)} bytes)")
        return pdf_bytes
    
    def export_chart_as_image(self, fig: go.Figure, format: str = 'png', width: int = 1200, height: int = 800) -> bytes:
        """
        Export Plotly chart as image
        
        Args:
            fig: Plotly figure object
            format: Image format ('png', 'jpeg', 'svg', 'pdf')
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Image as bytes
        """
        try:
            if format == 'svg':
                img_bytes = fig.to_image(format=format, width=width, height=height)
            else:
                img_bytes = fig.to_image(format=format, width=width, height=height, scale=2)
            
            logger.info(f"Chart exported as {format} ({len(img_bytes)} bytes)")
            return img_bytes
            
        except Exception as e:
            logger.error(f"Failed to export chart: {str(e)}")
            raise
    
    def create_market_overview_chart(self, analysis_data: Dict[str, str]) -> go.Figure:
        """Create market overview visualization from analysis data"""
        # Extract market size data if available
        market_size_text = analysis_data.get("market_size", "")
        
        # Create a default projection chart
        years = ['2024', '2025', '2026', '2027', '2028']
        
        # Try to extract actual numbers from market_size text
        market_values = self._extract_numbers_from_text(market_size_text, base_value=100, growth_rate=0.15)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=years,
            y=market_values,
            mode='lines+markers',
            name='Market Size',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=10, color='#ff7f0e')
        ))
        
        fig.update_layout(
            title='Market Size Projection (2024-2028)',
            xaxis_title='Year',
            yaxis_title='Market Size (Index)',
            template='plotly_white',
            hovermode='x unified',
            showlegend=True
        )
        
        return fig
    
    def create_competitor_analysis_chart(self, analysis_data: Dict[str, str]) -> go.Figure:
        """Create competitor analysis visualization"""
        competitors_text = analysis_data.get("competitors", "")
        
        # Extract competitor names (simple parsing)
        competitor_names = self._extract_competitor_names(competitors_text)
        
        if not competitor_names:
            competitor_names = ["Competitor A", "Competitor B", "Competitor C", "Your Company"]
        
        # Create sample metrics (in real scenario, extract from actual data)
        num_competitors = len(competitor_names)
        market_share = self._distribute_market_share(num_competitors)
        
        fig = go.Figure(data=[
            go.Bar(
                x=competitor_names,
                y=market_share,
                marker_color=['#2ca02c' if 'Your' in name else '#ff7f0e' for name in competitor_names],
                text=[f'{share}%' for share in market_share],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Market Share Distribution',
            xaxis_title='Companies',
            yaxis_title='Market Share (%)',
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def create_target_segment_chart(self, analysis_data: Dict[str, str]) -> go.Figure:
        """Create target market segment visualization"""
        segments_text = analysis_data.get("target_segments", "")
        
        # Extract segments
        segments = self._extract_segments(segments_text)
        
        if not segments:
            segments = {
                'Enterprise': 40,
                'SMB': 30,
                'Consumer': 25,
                'Government': 5
            }
        
        fig = go.Figure(data=[go.Pie(
            labels=list(segments.keys()),
            values=list(segments.values()),
            hole=0.3,
            marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#17becf'])
        )])
        
        fig.update_layout(
            title='Target Market Segmentation',
            template='plotly_white',
            showlegend=True
        )
        
        return fig
    
    def create_market_trends_chart(self, analysis_data: Dict[str, str]) -> go.Figure:
        """Create market trends visualization"""
        trends_text = analysis_data.get("market_trends", "")
        
        # Extract trend categories and their importance
        trends = self._extract_trends(trends_text)
        
        if not trends:
            trends = {
                'Digital Transformation': 85,
                'AI & Automation': 90,
                'Sustainability': 75,
                'Customer Experience': 80,
                'Data Analytics': 70
            }
        
        # Sort by importance
        sorted_trends = dict(sorted(trends.items(), key=lambda x: x[1], reverse=True))
        
        fig = go.Figure(data=[go.Bar(
            x=list(sorted_trends.values()),
            y=list(sorted_trends.keys()),
            orientation='h',
            marker_color='#1f77b4',
            text=[f'{val}%' for val in sorted_trends.values()],
            textposition='auto',
        )])
        
        fig.update_layout(
            title='Market Trends Impact Analysis',
            xaxis_title='Impact Score',
            yaxis_title='Trend Category',
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def _format_company_info(self, analysis_data: Dict[str, str]) -> str:
        """Format company information for PDF"""
        info_parts = []
        
        if analysis_data.get("company_name"):
            info_parts.append(f"<b>Company:</b> {analysis_data['company_name']}")
        
        if analysis_data.get("industry"):
            info_parts.append(f"<b>Industry:</b> {analysis_data['industry']}")
        
        if analysis_data.get("target_audience"):
            target = analysis_data['target_audience'][:200]
            info_parts.append(f"<b>Target Audience:</b> {target}...")
        
        if analysis_data.get("products_services"):
            products = analysis_data['products_services'][:200]
            info_parts.append(f"<b>Products/Services:</b> {products}...")
        
        if analysis_data.get("brand_description"):
            brand = analysis_data['brand_description'][:200]
            info_parts.append(f"<b>Brand:</b> {brand}...")
        
        return "<br/><br/>".join(info_parts)
    
    def _clean_markdown_for_pdf(self, text: str) -> str:
        """Clean and convert markdown text for PDF rendering"""
        if not text:
            return ""
        
        # Remove markdown headers and convert to bold
        text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Convert bold markdown
        text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
        
        # Convert italic markdown
        text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
        
        # Remove bullet points and replace with proper formatting
        text = re.sub(r'^[-*]\s+', '• ', text, flags=re.MULTILINE)
        
        # Remove extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Escape special HTML characters
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Restore our formatting tags
        text = text.replace('&lt;b&gt;', '<b>')
        text = text.replace('&lt;/b&gt;', '</b>')
        text = text.replace('&lt;i&gt;', '<i>')
        text = text.replace('&lt;/i&gt;', '</i>')
        
        return text
    
    def _extract_numbers_from_text(self, text: str, base_value: float = 100, growth_rate: float = 0.15) -> List[float]:
        """Extract numerical values from text or generate projections"""
        # Try to find numbers in text
        numbers = re.findall(r'[\d,.]+(?:\s*(?:billion|million|trillion))?', text, re.IGNORECASE)
        
        if numbers:
            # Parse found numbers
            values = []
            for num in numbers[:5]:  # Limit to 5 values
                try:
                    clean_num = re.sub(r'[^\d.]', '', num)
                    value = float(clean_num) if clean_num else base_value
                    values.append(value)
                except:
                    values.append(base_value)
            
            # If we have some values, use them; otherwise generate
            if len(values) >= 2:
                return values[:5] if len(values) >= 5 else values + [values[-1] * (1 + growth_rate) ** i for i in range(1, 6 - len(values))]
        
        # Generate default projections
        return [base_value * (1 + growth_rate) ** i for i in range(5)]
    
    def _extract_competitor_names(self, text: str) -> List[str]:
        """Extract competitor names from text"""
        if not text:
            return []
        
        # Simple extraction - look for capitalized words/phrases
        competitors = []
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                # Extract potential company name
                name = re.sub(r'^[-*•]\s*', '', line)
                name = re.sub(r':.*$', '', name)  # Remove description after colon
                name = name.strip()[:50]
                
                if name and len(name) > 3:
                    competitors.append(name)
        
        return competitors[:6]  # Limit to 6 competitors
    
    def _distribute_market_share(self, num_entities: int) -> List[int]:
        """Distribute market share among entities"""
        if num_entities == 0:
            return []
        
        # Create a realistic distribution
        shares = []
        remaining = 100
        
        for i in range(num_entities - 1):
            # First entity (your company) gets larger share
            if i == 0:
                share = min(35, remaining // 2)
            else:
                share = min(25, remaining // (num_entities - i))
            shares.append(share)
            remaining -= share
        
        # Last entity gets the remainder
        shares.append(remaining)
        
        return shares
    
    def _extract_segments(self, text: str) -> Dict[str, int]:
        """Extract market segments from text"""
        if not text:
            return {}
        
        segments = {}
        lines = text.split('\n')
        
        for line in lines:
            # Look for patterns like "Segment: XX%" or "- Segment (XX%)"
            match = re.search(r'([A-Za-z\s]+)[:\(]?\s*(\d+)%?', line)
            if match:
                segment_name = match.group(1).strip()
                segment_value = int(match.group(2))
                segments[segment_name] = segment_value
        
        return segments if segments else None
    
    def _extract_trends(self, text: str) -> Dict[str, int]:
        """Extract market trends from text"""
        if not text:
            return {}
        
        trends = {}
        lines = text.split('\n')
        
        for line in lines:
            # Look for trend mentions
            line = line.strip()
            if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                trend = re.sub(r'^[-*•]\s*', '', line)
                trend = trend.split(':')[0].strip()
                
                # Assign a score based on position (earlier = more important)
                if trend and len(trend) > 5:
                    score = max(50, 100 - len(trends) * 5)
                    trends[trend] = score
        
        return trends if trends else None
