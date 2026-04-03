"""
Market Intelligence Hub UI Components
Displays market analysis results and competitor insights
"""

import streamlit as st
import pandas as pd
from typing import Dict, Any, Optional
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import io


class MarketIntelligenceDashboard:
    """Market Intelligence Dashboard for displaying analysis results"""
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#17becf',
            'light': '#f0f0f0',
            'dark': '#2c3e50'
        }
    
    def display_market_overview(self, analysis_data: Dict[str, str]):
        """Display market overview section"""
        st.header("📊 Market Intelligence Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Industry",
                value=analysis_data.get("industry", "N/A"),
                help="Primary industry sector"
            )
        
        with col2:
            st.metric(
                label="Target Market",
                value=analysis_data.get("target_audience", "N/A")[:30] + "..." if len(analysis_data.get("target_audience", "")) > 30 else analysis_data.get("target_audience", "N/A"),
                help="Primary target audience"
            )
        
        with col3:
            st.metric(
                label="Analysis Date",
                value=datetime.now().strftime("%Y-%m-%d"),
                help="Date of market analysis"
            )
        
        # Company Information
        st.subheader("🏢 Company Profile")
        company_cols = st.columns(2)
        
        with company_cols[0]:
            st.write(f"**Company Name:** {analysis_data.get('company_name', 'N/A')}")
            st.write(f"**Products/Services:** {analysis_data.get('products_services', 'N/A')}")
        
        with company_cols[1]:
            st.write(f"**Brand Description:** {analysis_data.get('brand_description', 'N/A')}")
            st.write(f"**Marketing Goals:** {analysis_data.get('marketing_goals', 'N/A')}")
    
    def display_competitive_analysis(self, analysis_data: Dict[str, str]):
        """Display competitive analysis section"""
        st.header("⚔️ Competitive Landscape")
        
        competitors_text = analysis_data.get("competitors", "")
        if competitors_text and competitors_text != "Analysis failed":
            # Parse competitor information (this is a simplified parser)
            st.markdown(competitors_text)
            
            # Create a competitive positioning chart
            self._create_competitive_positioning_chart(analysis_data)
        else:
            st.info("Competitor analysis data not available. Run market analysis to generate insights.")
    
    def display_market_trends(self, analysis_data: Dict[str, str]):
        """Display market trends section"""
        st.header("📈 Market Trends & Opportunities")
        
        trends_text = analysis_data.get("market_trends", "")
        opportunities_text = analysis_data.get("market_opportunities", "")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔄 Current Trends")
            if trends_text and trends_text != "Analysis failed":
                st.markdown(trends_text)
            else:
                st.info("Market trends data not available.")
        
        with col2:
            st.subheader("🎯 Opportunities")
            if opportunities_text and opportunities_text != "Analysis failed":
                st.markdown(opportunities_text)
            else:
                st.info("Market opportunities data not available.")
    
    def display_target_segments(self, analysis_data: Dict[str, str]):
        """Display target segments analysis"""
        st.header("🎯 Target Market Segments")
        
        segments_text = analysis_data.get("target_segments", "")
        if segments_text and segments_text != "Analysis failed":
            st.markdown(segments_text)
            
            # Create a simple segmentation visualization
            self._create_segmentation_chart(analysis_data)
        else:
            st.info("Target segment analysis not available.")
    
    def display_competitive_advantages(self, analysis_data: Dict[str, str]):
        """Display competitive advantages"""
        st.header("🏆 Competitive Advantages")
        
        advantages_text = analysis_data.get("competitive_advantages", "")
        if advantages_text and advantages_text != "Analysis failed":
            st.markdown(advantages_text)
        else:
            st.info("Competitive advantages analysis not available.")
    
    def display_market_size(self, analysis_data: Dict[str, str]):
        """Display market size and growth potential"""
        st.header("📊 Market Size & Growth")
        
        market_size_text = analysis_data.get("market_size", "")
        if market_size_text and market_size_text != "Analysis failed":
            st.markdown(market_size_text)
            
            # Create market size visualization
            self._create_market_size_chart(analysis_data)
        else:
            st.info("Market size data not available.")
    
    def display_comprehensive_analysis(self, analysis_data: Dict[str, str]):
        """Display the comprehensive analysis report"""
        st.header("📋 Comprehensive Market Analysis")
        
        comprehensive_text = analysis_data.get("comprehensive_analysis", "")
        if comprehensive_text and comprehensive_text != "Analysis failed":
            st.markdown(comprehensive_text)
        else:
            st.info("Comprehensive analysis not available.")
    
    def _create_competitive_positioning_chart(self, analysis_data: Dict[str, str]):
        """Create competitive positioning chart"""
        # This is a mock chart - in reality, you'd parse the competitor data
        competitors = ["Your Company", "Competitor A", "Competitor B", "Competitor C"]
        market_share = [25, 35, 20, 20]  # Example data
        
        fig = go.Figure(data=[
            go.Bar(name='Market Share', x=competitors, y=market_share, 
                   marker_color=[self.colors['success'], self.colors['warning'], 
                                self.colors['warning'], self.colors['warning']])
        ])
        
        fig.update_layout(
            title='Market Share Comparison',
            xaxis_title='Companies',
            yaxis_title='Market Share (%)',
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_segmentation_chart(self, analysis_data: Dict[str, str]):
        """Create market segmentation pie chart"""
        # Mock segmentation data - would be parsed from analysis
        segments = ['Enterprise', 'SMB', 'Consumer', 'Government']
        sizes = [40, 30, 25, 5]
        
        fig = px.pie(values=sizes, names=segments, 
                     title='Target Market Segmentation',
                     color_discrete_map={
                         'Enterprise': self.colors['primary'],
                         'SMB': self.colors['secondary'],
                         'Consumer': self.colors['success'],
                         'Government': self.colors['info']
                     })
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_market_size_chart(self, analysis_data: Dict[str, str]):
        """Create market size growth projection chart"""
        # Mock data - would be parsed from analysis
        years = ['2024', '2025', '2026', '2027', '2028']
        market_size = [100, 120, 145, 175, 210]  # Billion USD
        
        fig = px.line(x=years, y=market_size, 
                      title='Market Size Projection',
                      labels={'x': 'Year', 'y': 'Market Size (Billion USD)'},
                      markers=True)
        
        fig.update_traces(line_color=self.colors['primary'], 
                         marker_color=self.colors['secondary'])
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_action_items(self, analysis_data: Dict[str, str]):
        """Display actionable insights and recommendations"""
        st.header("🎯 Action Items & Recommendations")
        
        # Extract recommendations from various sections
        recommendations = []
        
        # Parse opportunities for recommendations
        opportunities = analysis_data.get("market_opportunities", "")
        if opportunities and opportunities != "Analysis failed":
            # Simple extraction - in reality, you'd use more sophisticated parsing
            if "recommend" in opportunities.lower():
                recommendations.append("Focus on identified market opportunities")
        
        # Parse competitive advantages for recommendations
        advantages = analysis_data.get("competitive_advantages", "")
        if advantages and advantages != "Analysis failed":
            recommendations.append("Leverage identified competitive advantages")
        
        # Display recommendations
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.write(f"{i}. {rec}")
        else:
            st.info("Run market analysis to generate specific recommendations.")
    
    def display_full_dashboard(self, analysis_data: Optional[Dict[str, str]] = None):
        """Display the complete market intelligence dashboard"""
        st.title("🎯 Market Intelligence Hub")
        
        if not analysis_data:
            st.info("No market analysis data available. Please run market analysis first.")
            return
        
        # Check if analysis failed
        if analysis_data.get("error"):
            st.error(f"Market analysis failed: {analysis_data['error']}")
            return
        
        # Display all sections
        self.display_market_overview(analysis_data)
        
        st.markdown("---")
        self.display_competitive_analysis(analysis_data)
        
        st.markdown("---")
        self.display_market_trends(analysis_data)
        
        st.markdown("---")
        self.display_target_segments(analysis_data)
        
        st.markdown("---")
        self.display_competitive_advantages(analysis_data)
        
        st.markdown("---")
        self.display_market_size(analysis_data)
        
        st.markdown("---")
        self.display_action_items(analysis_data)
        
        # Export options
        st.markdown("---")
        self._display_export_options(analysis_data)
    
    def _display_export_options(self, analysis_data: Dict[str, str]):
        """Display export options for the analysis"""
        st.subheader("📤 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📄 Export to PDF", use_container_width=True):
                try:
                    from export_utils import MarketIntelligenceExporter
                    exporter = MarketIntelligenceExporter()
                    
                    # Generate HTML report with embedded charts (no Chrome dependency)
                    html_content = self._generate_html_report(analysis_data)
                    
                    # Provide download button for HTML
                    st.download_button(
                        label="⬇️ Download HTML Report",
                        data=html_content.encode('utf-8'),
                        file_name=f"market_intelligence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                        mime="text/html",
                        use_container_width=True
                    )
                    
                    st.success("✅ HTML report generated with interactive charts!")
                    
                    # Also try PDF if kaleido is available
                    try:
                        charts = []
                        
                        # Market overview chart
                        market_chart = exporter.create_market_overview_chart(analysis_data)
                        market_img = exporter.export_chart_as_image(market_chart, format='svg')
                        charts.append({'title': 'Market Size Projection', 'image_data': io.BytesIO(market_img)})
                        
                        # Competitor analysis chart
                        competitor_chart = exporter.create_competitor_analysis_chart(analysis_data)
                        competitor_img = exporter.export_chart_as_image(competitor_chart, format='svg')
                        charts.append({'title': 'Competitor Analysis', 'image_data': io.BytesIO(competitor_img)})
                        
                        # Target segment chart
                        segment_chart = exporter.create_target_segment_chart(analysis_data)
                        segment_img = exporter.export_chart_as_image(segment_chart, format='svg')
                        charts.append({'title': 'Target Segments', 'image_data': io.BytesIO(segment_img)})
                        
                        # Market trends chart
                        trends_chart = exporter.create_market_trends_chart(analysis_data)
                        trends_img = exporter.export_chart_as_image(trends_chart, format='svg')
                        charts.append({'title': 'Market Trends', 'image_data': io.BytesIO(trends_img)})
                        
                        # Generate PDF with SVG charts (vector quality)
                        pdf_bytes = exporter.export_to_pdf(analysis_data, charts)
                        
                        st.download_button(
                            label="⬇️ Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"market_intelligence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf",
                            use_container_width=True
                        )
                        
                        st.success("✅ Both HTML and PDF reports generated!")
                        
                    except Exception as pdf_error:
                        st.warning(f"PDF generation skipped (requires Chrome): Install Chrome or use HTML report")
                    
                except Exception as e:
                    st.error(f"Export failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        with col2:
            if st.button("📊 Export Charts", use_container_width=True):
                try:
                    from export_utils import MarketIntelligenceExporter
                    exporter = MarketIntelligenceExporter()
                    
                    # Create tabs for different charts
                    chart_tabs = st.tabs(["Market Overview", "Competitors", "Segments", "Trends"])
                    
                    # Market Overview Chart
                    with chart_tabs[0]:
                        try:
                            market_chart = exporter.create_market_overview_chart(analysis_data)
                            st.plotly_chart(market_chart, use_container_width=True, key='export_market')
                            
                            # Export as SVG (no Chrome needed)
                            market_svg = market_chart.to_image(format='svg', width=1200, height=800)
                            st.download_button(
                                label="⬇️ Download SVG",
                                data=market_svg,
                                file_name=f"market_overview_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                                mime="image/svg+xml",
                                use_container_width=True,
                                key='dl_market'
                            )
                        except Exception as e:
                            st.error(f"Market chart failed: {str(e)}")
                    
                    # Competitor Analysis Chart
                    with chart_tabs[1]:
                        try:
                            competitor_chart = exporter.create_competitor_analysis_chart(analysis_data)
                            st.plotly_chart(competitor_chart, use_container_width=True, key='export_competitor')
                            
                            competitor_svg = competitor_chart.to_image(format='svg', width=1200, height=800)
                            st.download_button(
                                label="⬇️ Download SVG",
                                data=competitor_svg,
                                file_name=f"competitor_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                                mime="image/svg+xml",
                                use_container_width=True,
                                key='dl_competitor'
                            )
                        except Exception as e:
                            st.error(f"Competitor chart failed: {str(e)}")
                    
                    # Target Segment Chart
                    with chart_tabs[2]:
                        try:
                            segment_chart = exporter.create_target_segment_chart(analysis_data)
                            st.plotly_chart(segment_chart, use_container_width=True, key='export_segment')
                            
                            segment_svg = segment_chart.to_image(format='svg', width=1200, height=800)
                            st.download_button(
                                label="⬇️ Download SVG",
                                data=segment_svg,
                                file_name=f"target_segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                                mime="image/svg+xml",
                                use_container_width=True,
                                key='dl_segment'
                            )
                        except Exception as e:
                            st.error(f"Segment chart failed: {str(e)}")
                    
                    # Market Trends Chart
                    with chart_tabs[3]:
                        try:
                            trends_chart = exporter.create_market_trends_chart(analysis_data)
                            st.plotly_chart(trends_chart, use_container_width=True, key='export_trends')
                            
                            trends_svg = trends_chart.to_image(format='svg', width=1200, height=800)
                            st.download_button(
                                label="⬇️ Download SVG",
                                data=trends_svg,
                                file_name=f"market_trends_{datetime.now().strftime('%Y%m%d_%H%M%S')}.svg",
                                mime="image/svg+xml",
                                use_container_width=True,
                                key='dl_trends'
                            )
                        except Exception as e:
                            st.error(f"Trends chart failed: {str(e)}")
                    
                except Exception as e:
                    st.error(f"Chart export failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        with col3:
            if st.button("📋 Export All Charts", use_container_width=True):
                try:
                    from export_utils import MarketIntelligenceExporter
                    exporter = MarketIntelligenceExporter()
                    
                    # Generate all charts and provide as zip
                    import zipfile
                    
                    zip_buffer = io.BytesIO()
                    
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                        # Market Overview
                        try:
                            market_chart = exporter.create_market_overview_chart(analysis_data)
                            market_svg = market_chart.to_image(format='svg', width=1200, height=800)
                            zip_file.writestr(f"market_overview.svg", market_svg)
                        except:
                            pass
                        
                        # Competitor Analysis
                        try:
                            competitor_chart = exporter.create_competitor_analysis_chart(analysis_data)
                            competitor_svg = competitor_chart.to_image(format='svg', width=1200, height=800)
                            zip_file.writestr(f"competitor_analysis.svg", competitor_svg)
                        except:
                            pass
                        
                        # Target Segments
                        try:
                            segment_chart = exporter.create_target_segment_chart(analysis_data)
                            segment_svg = segment_chart.to_image(format='svg', width=1200, height=800)
                            zip_file.writestr(f"target_segments.svg", segment_svg)
                        except:
                            pass
                        
                        # Market Trends
                        try:
                            trends_chart = exporter.create_market_trends_chart(analysis_data)
                            trends_svg = trends_chart.to_image(format='svg', width=1200, height=800)
                            zip_file.writestr(f"market_trends.svg", trends_svg)
                        except:
                            pass
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="⬇️ Download All Charts (ZIP)",
                        data=zip_buffer.getvalue(),
                        file_name=f"market_charts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                    
                    st.success("✅ All charts packaged!")
                    
                except Exception as e:
                    st.error(f"Batch export failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
    
    def _generate_html_report(self, analysis_data: Dict[str, str]) -> str:
        """Generate HTML report with embedded Plotly charts"""
        from export_utils import MarketIntelligenceExporter
        exporter = MarketIntelligenceExporter()
        
        # Generate charts as HTML divs
        charts_html = []
        
        try:
            # Market Overview
            market_chart = exporter.create_market_overview_chart(analysis_data)
            market_html = market_chart.to_html(full_html=False, include_plotlyjs='cdn')
            charts_html.append(f"<h2>Market Size Projection</h2>{market_html}")
        except:
            pass
        
        try:
            # Competitor Analysis
            competitor_chart = exporter.create_competitor_analysis_chart(analysis_data)
            competitor_html = competitor_chart.to_html(full_html=False, include_plotlyjs='cdn')
            charts_html.append(f"<h2>Competitor Analysis</h2>{competitor_html}")
        except:
            pass
        
        try:
            # Target Segments
            segment_chart = exporter.create_target_segment_chart(analysis_data)
            segment_html = segment_chart.to_html(full_html=False, include_plotlyjs='cdn')
            charts_html.append(f"<h2>Target Market Segmentation</h2>{segment_html}")
        except:
            pass
        
        try:
            # Market Trends
            trends_chart = exporter.create_market_trends_chart(analysis_data)
            trends_html = trends_chart.to_html(full_html=False, include_plotlyjs='cdn')
            charts_html.append(f"<h2>Market Trends Impact</h2>{trends_html}")
        except:
            pass
        
        # Build complete HTML report
        html_report = f"""<!DOCTYPE html>
<html>
<head>
    <title>Market Intelligence Report</title>
    <meta charset="UTF-8">
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #1f77b4;
            color: white;
            padding: 30px;
            text-align: center;
            border-radius: 5px;
            margin-bottom: 30px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }}
        .chart {{
            margin: 30px 0;
        }}
        h1 {{
            margin: 0;
        }}
        h2 {{
            color: #2c3e50;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 10px;
        }}
        .company-info {{
            line-height: 1.6;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Market Intelligence Report</h1>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>Company Overview</h2>
        <div class="company-info">
            {self._format_company_info_html(analysis_data)}
        </div>
    </div>
    
    {self._format_analysis_sections_html(analysis_data)}
    
    <div class="section">
        <h2>📈 Market Analysis Charts</h2>
        {''.join(charts_html)}
    </div>
    
    <div class="footer">
        <p>Report generated by MarketingAI Market Intelligence Hub</p>
    </div>
</body>
</html>"""
        
        return html_report
    
    def _format_company_info_html(self, analysis_data: Dict[str, str]) -> str:
        """Format company information for HTML report"""
        info_parts = []
        
        if analysis_data.get("company_name"):
            info_parts.append(f"<p><strong>Company:</strong> {analysis_data['company_name']}</p>")
        
        if analysis_data.get("industry"):
            info_parts.append(f"<p><strong>Industry:</strong> {analysis_data['industry']}</p>")
        
        if analysis_data.get("target_audience"):
            info_parts.append(f"<p><strong>Target Audience:</strong> {analysis_data['target_audience']}</p>")
        
        if analysis_data.get("products_services"):
            info_parts.append(f"<p><strong>Products/Services:</strong> {analysis_data['products_services']}</p>")
        
        if analysis_data.get("brand_description"):
            info_parts.append(f"<p><strong>Brand:</strong> {analysis_data['brand_description']}</p>")
        
        return '\n'.join(info_parts)
    
    def _format_analysis_sections_html(self, analysis_data: Dict[str, str]) -> str:
        """Format analysis sections for HTML report"""
        sections = [
            ("Competitive Landscape", "competitors"),
            ("Market Trends", "market_trends"),
            ("Market Opportunities", "market_opportunities"),
            ("Target Segments", "target_segments"),
            ("Competitive Advantages", "competitive_advantages"),
            ("Market Size & Growth", "market_size")
        ]
        
        html_sections = []
        
        for title, key in sections:
            if analysis_data.get(key) and analysis_data[key] != "Analysis failed":
                html_sections.append(f"""
    <div class="section">
        <h2>{title}</h2>
        <div style="white-space: pre-wrap; line-height: 1.6;">
            {analysis_data[key]}
        </div>
    </div>
                """)
        
        # Add comprehensive analysis if available
        if analysis_data.get("comprehensive_analysis") and analysis_data["comprehensive_analysis"] != "Analysis failed":
            html_sections.append(f"""
    <div class="section">
        <h2>Comprehensive Market Analysis</h2>
        <div style="white-space: pre-wrap; line-height: 1.6;">
            {analysis_data['comprehensive_analysis']}
        </div>
    </div>
            """)
        
        return '\n'.join(html_sections)


class MarketAnalysisWizard:
    """Wizard for step-by-step market analysis"""
    
    def __init__(self):
        self.dashboard = MarketIntelligenceDashboard()
    
    def run_analysis_wizard(self, business_context: Dict[str, str], llm):
        """Run the market analysis wizard"""
        st.header("🔍 Market Analysis Wizard")
        
        # Check if we have required business context
        required_fields = ['company_name', 'industry', 'target_audience', 'products_services']
        missing_fields = [field for field in required_fields if not business_context.get(field)]
        
        if missing_fields:
            st.warning(f"Missing required business information: {', '.join(missing_fields)}")
            st.info("Please complete the business setup first.")
            return None
        
        # Analysis options
        col1, col2 = st.columns(2)
        
        with col1:
            use_guided_research = st.checkbox(
                "Use Guided Research (Web Scraping)", 
                value=True,
                help="Enable web scraping for more comprehensive analysis"
            )
        
        with col2:
            analysis_depth = st.selectbox(
                "Analysis Depth",
                ["Basic", "Comprehensive", "Deep Dive"],
                help="Choose the depth of market analysis"
            )
        
        # Run analysis button
        if st.button("🚀 Run Market Analysis", type="primary"):
            with st.spinner("Conducting market analysis... This may take a few minutes."):
                try:
                    # Use dependency injection - analyzer should be passed in constructor
                    # or use a factory pattern to avoid circular imports
                    try:
                        from market_analyzer import MarketAnalyzer
                        analyzer = MarketAnalyzer()
                    except ImportError as e:
                        st.error(f"Failed to import MarketAnalyzer: {str(e)}")
                        return None
                    
                    # Run the analysis
                    analysis_results = analyzer.generate_guided_market_analysis_with_context(
                        llm=llm,
                        company_name=business_context['company_name'],
                        industry=business_context['industry'],
                        target_audience=business_context['target_audience'],
                        products_services=business_context['products_services'],
                        brand_description=business_context.get('brand_description', ''),
                        marketing_goals=business_context.get('marketing_goals', ''),
                        use_guided_research=use_guided_research
                    )
                    
                    # Store results in session state
                    st.session_state.market_analysis_results = analysis_results
                    
                    # USER JOURNEY FIX: Auto-update business context with new intelligence
                    if "business_context" in st.session_state:
                        updates_made = []
                        # Map analysis fields to context fields
                        for key, value in analysis_results.items():
                            # Only update if value exists and is meaningful
                            if value and isinstance(value, str) and value != "Analysis failed":
                                # Update if the field exists in business context
                                # (Note: analysis keys largely match business_context keys by design)
                                if key in st.session_state.business_context:
                                    # Optional: Check if we are overwriting substantial content? 
                                    # For now, we assume analysis is fresher/better.
                                    st.session_state.business_context[key] = value
                                    updates_made.append(key)
                        
                        if updates_made:
                            st.toast(f"✅ Updated {len(updates_made)} context fields with new insights!", icon="🔄")
                    
                    st.success("Market analysis completed successfully!")
                    
                    return analysis_results
                    
                except Exception as e:
                    st.error(f"Market analysis failed: {str(e)}")
                    return None
        
        return None
