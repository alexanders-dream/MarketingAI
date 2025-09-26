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
            if st.button("📄 Export to PDF"):
                st.info("PDF export functionality would be implemented here")
        
        with col2:
            if st.button("📊 Export Charts"):
                st.info("Chart export functionality would be implemented here")
        
        with col3:
            if st.button("📋 Copy Summary"):
                summary = self._generate_summary_text(analysis_data)
                st.code(summary)
                st.success("Summary copied to clipboard!")
    
    def _generate_summary_text(self, analysis_data: Dict[str, str]) -> str:
        """Generate a text summary of the analysis"""
        summary = f"""
Market Intelligence Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

Company: {analysis_data.get('company_name', 'N/A')}
Industry: {analysis_data.get('industry', 'N/A')}
Target Audience: {analysis_data.get('target_audience', 'N/A')}

Key Findings:
- Market Trends: {analysis_data.get('market_trends', 'N/A')[:200]}...
- Opportunities: {analysis_data.get('market_opportunities', 'N/A')[:200]}...
- Competitive Position: {analysis_data.get('competitive_advantages', 'N/A')[:200]}...

Recommended Actions:
1. Focus on identified market opportunities
2. Leverage competitive advantages
3. Monitor market trends for strategic positioning
"""
        return summary.strip()


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
                    
                    st.success("Market analysis completed successfully!")
                    
                    return analysis_results
                    
                except Exception as e:
                    st.error(f"Market analysis failed: {str(e)}")
                    return None
        
        return None
