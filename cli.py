#!/usr/bin/env python3
"""Command-line interface for the RAG system."""

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from document_indexer import DocumentIndexer
from vector_store import VectorStore
from document_retriever import DocumentRetriever
from llm_qa_system import LLMQASystem
from evaluator import RAGEvaluator, EvaluationSample
from chatbot import Chatbot

console = Console()


@click.group()
@click.option('--config', default='config.yaml', help='Path to configuration file')
@click.pass_context
def cli(ctx, config):
    """RAG System - Retrieval Augmented Generation CLI."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config


@cli.command()
@click.argument('source', type=click.Path(exists=True))
@click.option('--directory', '-d', is_flag=True, help='Source is a directory')
@click.option('--pattern', default='**/*.pdf', help='Glob pattern for directory loading')
@click.pass_context
def index(ctx, source, directory, pattern):
    """Index documents into the vector store (Q1 & Q2)."""
    config = ctx.obj['config']
    
    console.print(f"[bold blue]Indexing documents from: {source}[/]")
    
    # Initialize indexer
    indexer = DocumentIndexer(config)
    
    # Process documents
    chunks = indexer.process_documents(
        source=source,
        is_directory=directory,
        glob_pattern=pattern
    )
    
    # Create vector store
    vector_store = VectorStore(
        embeddings=indexer.get_embeddings_model(),
        config_path=config
    )
    vector_store.create_store(chunks)
    
    # Show stats
    stats = vector_store.get_collection_stats()
    console.print(f"[green]✓ Indexed {stats['count']} chunks[/]")
    console.print(f"[green]✓ Stored at: {stats['persist_directory']}[/]")


@cli.command()
@click.argument('query')
@click.option('--top-k', '-k', default=5, help='Number of results to return')
@click.pass_context
def search(ctx, query, top_k):
    """Search documents in the vector store (Q2)."""
    config = ctx.obj['config']
    
    console.print(f"[bold blue]Searching for: {query}[/]\n")
    
    retriever = DocumentRetriever(config_path=config)
    results = retriever.retrieve(query, top_k=top_k, with_scores=True)
    
    # Display results
    table = Table(title=f"Top {top_k} Results")
    table.add_column("Rank", style="cyan", width=6)
    table.add_column("Score", style="green", width=8)
    table.add_column("Source", style="yellow", width=20)
    table.add_column("Content", style="white", width=60)
    
    for r in results:
        content = r['content'][:100] + "..." if len(r['content']) > 100 else r['content']
        table.add_row(
            str(r['rank']),
            f"{r['score']:.4f}",
            r['metadata'].get('source', 'Unknown'),
            content
        )
    
    console.print(table)


@cli.command()
@click.argument('question')
@click.option('--show-sources', '-s', is_flag=True, help='Show source documents')
@click.pass_context
def ask(ctx, question, show_sources):
    """Ask a question using the RAG system (Q3)."""
    config = ctx.obj['config']
    
    console.print(f"[bold blue]Question: {question}[/]\n")
    console.print("[yellow]Generating answer...[/]\n")
    
    qa_system = LLMQASystem(config)
    result = qa_system.answer(question, return_sources=show_sources)
    
    console.print("[bold green]Answer:[/]")
    console.print(result['answer'])
    
    if show_sources and 'sources' in result:
        console.print("\n[bold yellow]Sources:[/]")
        for src in result['sources']:
            console.print(f"  • {src['source']} (page {src['page']}) - score: {src['score']:.4f}")


@cli.command()
@click.pass_context
def chat(ctx):
    """Start interactive chatbot mode (Q5)."""
    config = ctx.obj['config']
    
    console.print("[bold blue]Starting RAG Chatbot...[/]\n")
    
    chatbot = Chatbot(config)
    chatbot.interactive_mode()


@cli.command()
@click.option('--output', '-o', default='evaluation_results.json', help='Output file path')
@click.pass_context
def evaluate(ctx, output):
    """Run evaluation on the RAG system (Q4)."""
    config = ctx.obj['config']
    
    console.print("[bold blue]Running RAG Evaluation...[/]\n")
    
    # Initialize QA system and evaluator
    qa_system = LLMQASystem(config)
    evaluator = RAGEvaluator(qa_system=qa_system, config_path=config)
    
    # Example evaluation dataset - replace with your test data
    test_data = evaluator.create_evaluation_dataset(
        questions=[
            "What is the main topic of the documents?",
            "Can you summarize the key findings?",
        ],
        ground_truths=[
            "The main topic is...",  # Replace with actual ground truth
            "The key findings are...",  # Replace with actual ground truth
        ],
        expected_sources=[
            ["document1.pdf"],  # Replace with actual sources
            ["document1.pdf", "document2.pdf"],
        ]
    )
    
    # Run evaluation
    results = evaluator.evaluate_dataset(test_data)
    evaluator.print_summary()
    evaluator.save_results(output)
    
    console.print(f"\n[green]✓ Results saved to: {output}[/]")


@cli.command()
@click.pass_context
def stats(ctx):
    """Show vector store statistics."""
    config = ctx.obj['config']
    
    indexer = DocumentIndexer(config)
    vector_store = VectorStore(
        embeddings=indexer.get_embeddings_model(),
        config_path=config
    )
    
    try:
        stats = vector_store.get_collection_stats()
        
        table = Table(title="Vector Store Statistics")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            table.add_row(key, str(value))
        
        console.print(table)
    except FileNotFoundError:
        console.print("[red]Vector store not found. Run 'index' command first.[/]")


if __name__ == '__main__':
    cli()
