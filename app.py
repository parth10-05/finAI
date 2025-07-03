import streamlit as st
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import pandas as pd
from typing import Dict, List
import io  

# Set page config
st.set_page_config(
    page_title="Financial Research Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS with modern design (keep all your existing CSS)

# Initialize session state for chat
if "messages" not in st.session_state:
    st.session_state.messages = []
if 'query' not in st.session_state:
    st.session_state.query = ""
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""
if 'active_tab' not in st.session_state:
    st.session_state.active_tab = "Research Agent"  # Default to Research Agent tab

# Agent factory functions (keeping all original logic)
def create_financial_agent(api_key: str):
    return Agent(
        name="Financial Analyst Pro",
        model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct", api_key=api_key),
        tools=[YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_info=True,
            company_news=True
        )],
        show_tool_calls=True,
        markdown=True,
        instructions=[
            "You are a senior financial analyst with 20+ years of market experience",
            "For any stock analysis, always include:",
            "1. Current price and 52-week range",
            "2. Market capitalization",
            "3. Key ratios (P/E, P/S, Debt/Equity, ROE)",
            "4. Revenue and earnings growth (YoY and QoQ)",
            "5. Analyst price targets and recommendations",
            "",
            "Format all numbers properly:",
            "- Currency: $1.2B (not 1200000000)",
            "- Percentages: 5.3% (not 0.053)",
            "- Ratios: 12.5x (not 12.5)",
            "",
            "Create comparison tables for:",
            "- Valuation metrics",
            "- Growth rates",
            "- Profitability"
        ]
    )

def create_web_researcher(api_key: str):
    return Agent(
        name="Market Intelligence Specialist",
        model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct", api_key=api_key),
        tools=[DuckDuckGo()],
        show_tool_calls=True,
        markdown=True,
        instructions=[
            "You are a professional market researcher with access to real-time web data",
            "Always verify information from multiple sources",
            "Prioritize data from: Bloomberg, Reuters, WSJ, SEC filings, and company websites",
            "Include publication dates for all sourced information",
            "Format news with headlines, dates, and key points",
            "For market trends, identify:",
            "  - Key drivers and catalysts",
            "  - Major players and their market share",
            "  - Regulatory and macroeconomic factors",
            "Always end with source reliability assessment"
        ]
    )

def create_agents_team(api_key: str):
    return Agent(
        team=[create_financial_agent(api_key), create_web_researcher(api_key)],
        model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct", api_key=api_key),
        show_tool_calls=True,
        markdown=True,
        debug_mode=True,
        instructions=[
            "COLLABORATION PROTOCOL:",
            "1. Financial analyst handles all quantitative data and ratios",
            "2. Web researcher provides qualitative context and news",
            "3. Both agents cross-validate findings before final output",
            "",
            "OUTPUT REQUIREMENTS:",
            "- Start with executive summary (3-5 bullet points)",
            "- Include both fundamental and technical analysis",
            "- Present data in this order:",
            "  1. Company overview",
            "  2. Financial health assessment",
            "  3. Growth prospects",
            "  4. Competitive positioning",
            "  5. Risk factors",
            "- All tables must include:",
            "  - Timeframe reference",
            "  - Data source",
            "  - Calculation methodology when non-standard",
            "- Conclude with investment thesis and price targets if available"
        ]
    )

def create_financial_chatbot(api_key: str):
    return Agent(
        name="Financial Chatbot",
        model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct", api_key=api_key),
        tools=[YFinanceTools(
            stock_price=True,
            analyst_recommendations=True,
            stock_fundamentals=True,
            company_info=True,
            company_news=True
        ), DuckDuckGo()],
        show_tool_calls=True,
        markdown=True,
        instructions=[
            "You are a friendly financial assistant that helps users with investment research and market analysis",
            "Your responses should be conversational but professional",
            "When answering questions:",
            "1. First understand what the user is asking",
            "2. Provide clear, concise explanations of financial concepts when needed",
            "3. Use bullet points or numbered lists for complex information",
            "4. Always cite sources for factual information",
            "5. Offer to provide more details if the user seems interested",
            "",
            "For stock-related questions, always include:",
            "- Current price and key metrics",
            "- Recent performance",
            "- Any relevant news",
            "",
            "For general financial questions:",
            "- Explain concepts simply first",
            "- Then provide more technical details if appropriate",
            "- Use analogies when helpful",
            "",
            "Maintain a helpful, professional tone throughout the conversation"
        ]
    )

def extract_tables_from_markdown(markdown_text: str) -> Dict[str, pd.DataFrame]:
    """Extract tables from markdown text and return as DataFrames"""
    tables = {}
    lines = markdown_text.split('\n')
    table_lines = []
    in_table = False
    table_name = "Table"
    table_count = 1
    
    for line in lines:
        if line.startswith('|') and '---' not in line:
            in_table = True
            table_lines.append(line)
        elif in_table and not line.startswith('|'):
            in_table = False
            if table_lines:
                try:
                    content = '\n'.join(table_lines)
                    df = pd.read_csv(io.StringIO(content), sep='|', skipinitialspace=True)
                    df = df.dropna(axis=1, how='all')
                    df.columns = df.columns.str.strip()
                    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
                    tables[f"{table_name} {table_count}"] = df
                    table_count += 1
                except Exception as e:
                    st.warning(f"Couldn't parse table: {e}")
                table_lines = []
    
    return tables

# Enhanced header
st.markdown("""
    <div class="custom-header">
        <h1>📈 Financial Research Agent</h1>
        <p>Advanced AI-powered financial analysis and market research platform</p>
    </div>
""", unsafe_allow_html=True)

# Tab selection
tab1, tab2 = st.tabs(["🔍 Research Agent", "💬 Financial Chatbot"])

with tab1:
    # Research Agent tab content (your existing code)
    # Enhanced sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        # API Key input with improved styling
        st.session_state.groq_api_key = st.text_input(
            "🔑 Groq API Key",
            type="password",
            value=st.session_state.groq_api_key,
            help="Get your API key from https://console.groq.com/keys",
            placeholder="Enter your API key here..."
        )
        
        st.markdown("### 🎛️ Options")
        show_raw_response = st.checkbox("📝 Show raw agent response", value=False)
        show_tables_separately = st.checkbox("📊 Show extracted tables separately", value=True)
        
        st.markdown("---")
        
        # Agent team info with feature cards
        st.markdown("### 🤖 AI Agent Team")
        
        st.markdown("""
            <div class="feature-card">
                <h4>💹 Financial Analyst Pro</h4>
                <p>Expert in quantitative analysis, financial metrics, and market data</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h4>🌐 Market Intelligence Specialist</h4>
                <p>Real-time web research and qualitative market insights</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Capabilities section
        st.markdown("### ✨ Capabilities")
        capabilities = [
            "📈 Stock price analysis",
            "📊 Financial fundamentals",
            "🎯 Analyst recommendations",
            "📰 Real-time news & trends",
            "🔍 Market research",
            "📋 Comparative analysis"
        ]
        
        for capability in capabilities:
            st.markdown(f"• {capability}")
        
        st.markdown("---")
        
        # Quick stats
        st.markdown("### 📊 Quick Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div class="stats-card">
                    <h3>2</h3>
                    <p>AI Agents</p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="stats-card">
                    <h3>∞</h3>
                    <p>Data Sources</p>
                </div>
            """, unsafe_allow_html=True)

    # Main content area for Research Agent
    st.markdown("### 💬 Ask Your Financial Question")

    # Enhanced query input
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "",
            placeholder="e.g., Compare Apple and Microsoft financials or analyze Tesla's latest earnings...",
            key="main_query_input",
            value=st.session_state.query,
            label_visibility="collapsed"
        )
    with col2:
        st.write("")  # Spacing
        run_query = st.button("🚀 Analyze", key="run_button")

    # Enhanced example queries section
    st.markdown("### 💡 Example Queries")

    def set_example_query(example_query):
        st.session_state.query = example_query
        st.rerun()

    # Create example buttons in a more attractive layout
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🍎 Compare Apple & Microsoft", key="ex1", help="Deep dive into both companies' financials"):
            set_example_query("Compare Apple and Microsoft financials")

    with col2:
        if st.button("⚡ Tesla Latest News", key="ex2", help="Get the most recent Tesla developments"):
            set_example_query("What are the latest news and developments about Tesla?")

    with col3:
        if st.button("📊 S&P 500 Analysis", key="ex3", help="Comprehensive market overview"):
            set_example_query("Analyze current S&P 500 trends with key statistics")

    # Additional example queries
    st.markdown("#### More Examples")
    col4, col5, col6 = st.columns(3)

    with col4:
        if st.button("🏦 Banking Sector Trends", key="ex4", help="Analyze the banking industry"):
            set_example_query("Analyze current banking sector trends and top performing bank stocks")

    with col5:
        if st.button("🚗 EV Industry", key="ex5", help="Electric vehicle market analysis"):
            set_example_query("Analyze the electric vehicle industry including major automakers and battery companies")

    with col6:
        if st.button("🏠 Real Estate Stocks", key="ex6", help="REITs and real estate analysis"):
            set_example_query("Analyze top performing REITs")

    # Handle query execution with enhanced UI
    if run_query and query:
        if not st.session_state.groq_api_key:
            st.error("🔑 Please enter your Groq API key in the sidebar to continue")
            st.stop()
            
        # Enhanced loading state
        with st.spinner("🧠 Our AI agents are analyzing your query..."):
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            try:
                progress_bar.progress(25)
                progress_text.text("🔄 Initializing agent team...")
                
                agents_team = create_agents_team(st.session_state.groq_api_key)
                
                progress_bar.progress(50)
                progress_text.text("📊 Gathering financial data...")
                
                response = agents_team.run(query)
                
                progress_bar.progress(75)
                progress_text.text("🔍 Processing results...")
                
                progress_bar.progress(100)
                progress_text.text("✅ Analysis complete!")
                
                # Clear progress indicators
                progress_bar.empty()
                progress_text.empty()
                
                if response and response.content:
                    # Show raw response in expander if requested
                    if show_raw_response:
                        with st.expander("🔍 Full Agent Response Details"):
                            st.markdown(response.content)
                    
                    # Main results section
                    st.markdown("### 📋 Analysis Results")
                    
                    # Enhanced response display
                    st.markdown(
                        f'<div class="agent-response">{response.content}</div>', 
                        unsafe_allow_html=True
                    )
                    
                    # Enhanced tables section
                    if show_tables_separately:
                        tables = extract_tables_from_markdown(response.content)
                        if tables:
                            st.markdown("### 📊 Data Tables")
                            
                            # Create tabs for multiple tables
                            if len(tables) > 1:
                                tab_names = list(tables.keys())
                                tabs = st.tabs(tab_names)
                                
                                for i, (name, df) in enumerate(tables.items()):
                                    with tabs[i]:
                                        st.dataframe(df, use_container_width=True)
                                        csv = df.to_csv(index=False).encode('utf-8')
                                        st.download_button(
                                            label=f"📥 Download {name} as CSV",
                                            data=csv,
                                            file_name=f"{name.lower().replace(' ', '_')}.csv",
                                            mime='text/csv',
                                            key=f"download_{name}",
                                            help=f"Download {name} data as CSV file"
                                        )
                            else:
                                # Single table display
                                for name, df in tables.items():
                                    st.markdown(f"#### {name}")
                                    st.dataframe(df, use_container_width=True)
                                    csv = df.to_csv(index=False).encode('utf-8')
                                    st.download_button(
                                        label=f"📥 Download {name} as CSV",
                                        data=csv,
                                        file_name=f"{name.lower().replace(' ', '_')}.csv",
                                        mime='text/csv',
                                        key=f"download_{name}",
                                        help=f"Download {name} data as CSV file"
                                    )
                    
                    # Success message
                    st.success("✅ Analysis completed successfully! Your financial research is ready.")
                    
                else:
                    st.warning("⚠️ No response received from the agent. Please try again with a different query.")
                    
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")
                st.error("Please check your API key and try again. If the problem persists, try a different query.")
                
    elif run_query and not query:
        st.warning("⚠️ Please enter a financial question before clicking Analyze.")

with tab2:
    # Financial Chatbot tab content
    st.markdown("### 💬 Financial Chatbot")
    st.markdown("Ask any financial question and get expert answers in a conversational format.")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a financial question..."):
        if not st.session_state.groq_api_key:
            st.error("🔑 Please enter your Groq API key in the sidebar to continue")
            st.stop()
            
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            try:
                # Create chatbot instance
                chatbot = create_financial_chatbot(st.session_state.groq_api_key)
                
                # Get response from chatbot
                response = chatbot.run(prompt)
                
                if response and response.content:
                    # Display the response
                    full_response = response.content
                    message_placeholder.markdown(full_response)
                else:
                    full_response = "I couldn't generate a response. Please try again with a different question."
                    message_placeholder.markdown(full_response)
                    
            except Exception as e:
                full_response = f"An error occurred: {str(e)}"
                message_placeholder.error(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})

# Enhanced footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #718096; font-size: 0.9rem; margin-top: 2rem;">
        <p>🚀 Powered by AI • Built with Streamlit • Enhanced Financial Intelligence</p>
        <p>💡 <strong>Pro Tip:</strong> Be specific in your queries for better results!</p>
    </div>
""", unsafe_allow_html=True)