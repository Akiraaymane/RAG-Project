#!/usr/bin/env python3
"""
cli.py - RAG System CLI avec Rich et Click
Interface interactive pour indexation, recherche, questions et évaluation
"""

import sys
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import time

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt

from dotenv import load_dotenv

from src.indexer import DocumentIndexer
from src.retriever import DocumentRetriever
from src.system_qa import QASystem
from src.evaluator import RAGEvaluator, EvaluationSample

load_dotenv()

console = Console()

# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                            BANNER & UTILITIES                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

BANNER = """
[bold cyan]
  ╔═══════════════════════════════════════════════════════╗
  ║                                                       ║
  ║           [bold yellow]RAG AYMANE[/]                        ║
  ║                                                       ║
  ║  [dim]Retrieval Augmented Generation - Philosophie[/]    ║
  ║                                                       ║
  ╚═══════════════════════════════════════════════════════╝
[/bold cyan]
"""

def print_banner():
    """Affiche la bannière"""
    console.print(BANNER)


def create_results_table(results: List, title: str = "Résultats") -> Table:
    """Crée une table formatée pour les résultats"""
    table = Table(
        title=f"🔍 {title}",
        show_header=True,
        header_style="bold blue",
        show_lines=True
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Score", style="green", width=8)
    table.add_column("Source", style="yellow", width=25)
    table.add_column("Page", style="cyan", width=6)
    table.add_column("Contenu", style="white", max_width=50)
    
    for i, (doc, score) in enumerate(results, 1):
        content = doc.page_content[:80].replace('\n', ' ')
        if len(doc.page_content) > 80:
            content += "..."
        
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'N/A')
        
        table.add_row(str(i), f"{score:.4f}", source, str(page), content)
    
    return table


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                              MAIN CLI GROUP                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@click.group()
@click.version_option(version='1.0.0', prog_name='RAG System')
def cli():
    """
    🤖 RAG System - Retrieval Augmented Generation CLI
    
    Un système puissant pour répondre à des questions sur des documents PDF
    en utilisant la recherche vectorielle et des LLMs locaux.
    """
    pass


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                           Q1: INDEX COMMAND                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@cli.command()
def index():
    """📚 Indexer les documents PDF du dossier data/"""
    print_banner()
    
    console.print(Panel(
        "[bold]Indexation des documents[/]\n"
        "Source: [cyan]data/[/]",
        title="📚 Document Indexer",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Chargement du modèle...", total=None)
        
        try:
            indexer = DocumentIndexer()
            progress.update(task, description="[green]✓ Modèle chargé")
            
            progress.update(task, description="[cyan]Traitement des documents...")
            stats = indexer.index(clear=False)
            progress.update(task, description="[green]✓ Documents traités")
        except Exception as e:
            console.print(f"[red]❌ Erreur: {str(e)}[/]")
            return
    
    # Afficher les stats
    console.print()
    table = Table(
        title="✅ Statistiques d'indexation",
        show_header=True,
        header_style="bold green"
    )
    table.add_column("Source", style="cyan")
    table.add_column("Chunks", style="green")
    
    for source, count in stats['chunks_by_source'].items():
        table.add_row(source, str(count))
    
    console.print(table)
    
    console.print(Panel(
        f"[green]Total: {stats['total_chunks']} chunks indexés![/]",
        border_style="green"
    ))


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                           Q2: SEARCH COMMAND                              ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@cli.command()
@click.argument('query')
@click.option('--top-k', '-k', default=3, help='Nombre de résultats')
def search(query, top_k):
    """🔍 Rechercher dans la base vectorielle"""
    print_banner()
    
    console.print(Panel(
        f"[bold]Requête:[/] [cyan]{query}[/]",
        title="🔍 Recherche Sémantique",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Recherche en cours...", total=None)
        
        try:
            retriever = DocumentRetriever()
            retriever.load_vector_store()
            results = retriever.search(query, top_k=top_k)
            progress.update(task, description=f"[green]✓ {len(results)} résultat(s) trouvé(s)")
        except Exception as e:
            console.print(f"[red]❌ Erreur: {str(e)}[/]")
            return
    
    if results:
        console.print()
        console.print(create_results_table(results, "Résultats de Recherche"))
    else:
        console.print(Panel(
            "[yellow]⚠️ Aucun résultat trouvé[/]",
            border_style="yellow"
        ))


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                            Q3: ASK COMMAND                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@cli.command()
@click.argument('question')
@click.option('--top-k', '-k', default=3, help='Documents à récupérer')
@click.option('--sources', '-s', is_flag=True, help='Afficher les sources')
def ask(question, top_k, sources):
    """❓ Poser une question au système QA"""
    print_banner()
    
    console.print(Panel(
        f"[bold]Question:[/] [cyan]{question}[/]",
        title="❓ Question-Réponse",
        border_style="blue"
    ))
    
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        console.print("[red]❌ Erreur: Variable HF_API_KEY non définie[/]")
        console.print("Définissez-la dans le fichier .env")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Initialisation...", total=None)
        
        try:
            qa_system = QASystem(hf_api_key=hf_api_key)
            
            progress.update(task, description="[cyan]Récupération du contexte...")
            progress.update(task, description="[cyan]Génération de la réponse...")
            
            result = qa_system.query(question, top_k=top_k)
            progress.update(task, description="[green]✓ Réponse générée")
        except Exception as e:
            console.print(f"[red]❌ Erreur: {str(e)}[/]")
            return
    
    # Afficher la réponse
    console.print()
    console.print(Panel(
        Markdown(result['answer']),
        title="💡 Réponse",
        border_style="green",
        padding=(1, 2)
    ))
    
    # Afficher les sources si demandé
    if sources and result.get('sources'):
        console.print()
        table = Table(
            title="📚 Sources Utilisées",
            show_header=True,
            header_style="bold yellow"
        )
        table.add_column("Document", style="cyan")
        table.add_column("Statut", style="green")
        
        for src in result['sources']:
            table.add_row(src, "✓")
        
        console.print(table)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                       Q4: EVALUATE COMMANDS                               ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@cli.group(name='evaluate')
def evaluate_group():
    """📊 Commandes d'évaluation du système RAG"""
    pass


@evaluate_group.command(name='sample')
@click.argument('question')
@click.option('--ground-truth', '-g', required=True, help='Réponse attendue')
@click.option('--top-k', '-k', default=3, help='Documents à récupérer')
def evaluate_sample(question, ground_truth, top_k):
    """
    📊 Évaluer une seule question
    
    Exemple:
        python cli.py evaluate sample "Qui a fondé le café philo?" \\
            --ground-truth "Marc Sautet" --top-k 3
    """
    print_banner()
    
    console.print(Panel(
        f"[bold]Question:[/] [cyan]{question}[/]\n"
        f"[bold]Réponse attendue:[/] [cyan]{ground_truth}[/]",
        title="📊 Évaluation",
        border_style="blue"
    ))
    
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        console.print("[red]❌ Erreur: Variable HF_API_KEY non définie[/]")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Évaluation en cours...", total=None)
        
        try:
            qa_system = QASystem(hf_api_key=hf_api_key)
            evaluator = RAGEvaluator(qa_system)
            
            sample = EvaluationSample(
                question=question,
                ground_truth=ground_truth,
                expected_sources=[]
            )
            
            result = evaluator.evaluate_sample(sample, k=top_k)
            progress.update(task, description="[green]✓ Évaluation complète")
        except Exception as e:
            console.print(f"[red]❌ Erreur: {str(e)}[/]")
            return
    
    # Afficher les résultats
    console.print()
    console.print(Panel(
        f"[bold]Question:[/] {result.question}\n"
        f"[bold]Réponse générée:[/] {result.generated_answer}\n"
        f"[bold]Latence:[/] {result.latency_ms:.2f}ms",
        title="📋 Détails",
        border_style="cyan"
    ))
    
    # Métriques de retrieval
    console.print()
    ret_table = Table(
        title="🔍 Métriques de Retrieval",
        show_header=True,
        header_style="bold blue"
    )
    ret_table.add_column("Métrique", style="cyan")
    ret_table.add_column("Score", style="green")
    ret_table.add_column("Barre", style="white")
    
    for metric, value in result.retrieval_metrics.items():
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        ret_table.add_row(metric, f"{value:.4f}", bar)
    
    console.print(ret_table)
    
    # Métriques de réponse
    console.print()
    ans_table = Table(
        title="💡 Métriques de Réponse",
        show_header=True,
        header_style="bold green"
    )
    ans_table.add_column("Métrique", style="cyan")
    ans_table.add_column("Score", style="green")
    ans_table.add_column("Barre", style="white")
    
    for metric, value in result.answer_metrics.items():
        bar = "█" * int(value * 20) + "░" * (20 - int(value * 20))
        ans_table.add_row(metric, f"{value:.4f}", bar)
    
    console.print(ans_table)


@evaluate_group.command(name='dataset')
@click.option('--dataset', '-d', default='data/evaluation_dataset.json', 
              help='Fichier du dataset')
@click.option('--output', '-o', default='results/evaluation_results.json', 
              help='Fichier de sortie')
@click.option('--top-k', '-k', default=3, help='Documents à récupérer')
def evaluate_dataset(dataset, output, top_k):
    """
    📊 Évaluer un dataset complet
    
    Exemple:
        python cli.py evaluate dataset --dataset data/evaluation_dataset.json \\
            --output results/eval.json --top-k 3
    """
    print_banner()
    
    if not Path(dataset).exists():
        console.print(f"[red]❌ Dataset non trouvé: {dataset}[/]")
        return
    
    console.print(Panel(
        f"[bold]Dataset:[/] [cyan]{dataset}[/]\n"
        f"[bold]Résultats:[/] [cyan]{output}[/]",
        title="📊 Évaluation Dataset",
        border_style="blue"
    ))
    
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        console.print("[red]❌ Erreur: Variable HF_API_KEY non définie[/]")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("[cyan]Initialisation...", total=None)
        
        try:
            qa_system = QASystem(hf_api_key=hf_api_key)
            evaluator = RAGEvaluator(qa_system)
            
            progress.update(task, description="[cyan]Chargement du dataset...")
            test_data = evaluator.load_dataset(dataset)
            progress.update(task, description=f"[green]✓ {len(test_data)} échantillons chargés")
            
            progress.update(task, description="[cyan]Évaluation...")
            results = evaluator.evaluate_dataset(test_data, k=top_k, verbose=True)
            progress.update(task, description="[green]✓ Évaluation complète")
            
            progress.update(task, description="[cyan]Sauvegarde...")
            evaluator.save_results(output)
            progress.update(task, description="[green]✓ Résultats sauvegardés")
        except Exception as e:
            console.print(f"[red]❌ Erreur: {str(e)}[/]")
            return
    
    # Afficher le résumé
    console.print()
    evaluator.print_summary()
    
    console.print(Panel(
        f"[green]✅ Résultats sauvegardés dans:[/]\n"
        f"[cyan]{output}[/]",
        border_style="green"
    ))


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                            HELP COMMAND                                   ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@cli.command(name='help')
def show_help():
    """📖 Afficher l'aide complète"""
    print_banner()
    
    console.print(Panel(
        """[bold cyan]COMMANDES DISPONIBLES:[/]

[bold]📚 Indexation:[/]
  [cyan]python cli.py index[/]
    → Indexer tous les documents PDF

[bold]🔍 Recherche:[/]
  [cyan]python cli.py search "<query>" [-k 5][/]
    → Rechercher dans la base vectorielle
    [dim]Exemple: python cli.py search "philosophie" -k 3[/]

[bold]❓ Question-Réponse:[/]
  [cyan]python cli.py ask "<question>" [-k 3] [-s][/]
    → Poser une question au système
    [dim]Exemple: python cli.py ask "Qui a fondé le café philo?" -s[/]

[bold]📊 Évaluation:[/]
  [cyan]python cli.py evaluate sample "<question>" -g "<réponse>" [-k 3][/]
    → Évaluer une question
    
  [cyan]python cli.py evaluate dataset [-d <fichier>] [-o <fichier>] [-k 3][/]
    → Évaluer un dataset complet

[bold]ℹ️ Options:[/]
  [cyan]-k, --top-k[/]      Nombre de documents (défaut: 3)
  [cyan]-s, --sources[/]    Afficher les sources (ask)
  [cyan]-g, --ground-truth[/] Réponse attendue (evaluate sample)
  [cyan]-d, --dataset[/]    Fichier du dataset (evaluate dataset)
  [cyan]-o, --output[/]     Fichier de sortie (evaluate dataset)
""",
        title="📖 Aide",
        border_style="blue"
    ))


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                              ENTRY POINT                                  ║
# ╚════════════════════════════════════════════════════════════════════════════╝

if __name__ == '__main__':
    cli()