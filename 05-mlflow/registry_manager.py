#!/usr/bin/env python3
"""
MLflow Registry Manager.
Gerencia ciclo de vida dos modelos: Staging → Production → Archived.
"""

import os
import mlflow
from mlflow.tracking import MlflowClient
from typing import Optional, Dict, List

MLFLOW_TRACKING_URI = "http://mlflow-server:5000"
MODEL_NAME = "agent-error-predictor"

class RegistryManager:
    """
    Gerencia o ciclo de vida de modelos no MLflow Model Registry.
    """
    
    def __init__(self, tracking_uri: str = MLFLOW_TRACKING_URI, model_name: str = MODEL_NAME):
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
        self.model_name = model_name
        
        # Garantir que o modelo existe no registry
        self._ensure_registered_model()
    
    def _ensure_registered_model(self):
        """Cria o modelo no registry se não existir."""
        try:
            self.client.get_registered_model(self.model_name)
        except mlflow.exceptions.MlflowException:
            self.client.create_registered_model(self.model_name)
            print(f"✅ Modelo '{self.model_name}' criado no registry")
    
    def list_versions(self, stage: Optional[str] = None) -> List[Dict]:
        """
        Lista versões do modelo. Filtra por stage se especificado.
        """
        versions = self.client.search_model_versions(f"name='{self.model_name}'")
        result = []
        for v in versions:
            info = {
                'version': v.version,
                'stage': v.current_stage,
                'run_id': v.run_id,
                'status': v.status,
                'creation_timestamp': v.creation_timestamp,
            }
            if stage is None or v.current_stage == stage:
                result.append(info)
        return result
    
    def get_production_version(self) -> Optional[Dict]:
        """Retorna a versão atual em Production."""
        versions = self.list_versions(stage="Production")
        return versions[0] if versions else None
    
    def get_staging_version(self) -> Optional[Dict]:
        """Retorna a versão atual em Staging."""
        versions = self.list_versions(stage="Staging")
        return versions[0] if versions else None
    
    def promote_to_staging(self, run_id: str) -> str:
        """
        Registra uma nova versão do modelo e move para Staging.
        """
        # Criar nova versão
        model_version = self.client.create_model_version(
            name=self.model_name,
            source=f"runs:/{run_id}/model",
            run_id=run_id
        )
        
        # Mover para Staging
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        print(f"✅ Modelo v{model_version.version} promovido para Staging (run_id={run_id})")
        return model_version.version
    
    def promote_to_production(self, version: str) -> bool:
        """
        Promove uma versão de Staging para Production.
        Arquiva a versão anterior em Production.
        """
        # Arquivar versão atual em Production
        current_prod = self.get_production_version()
        if current_prod:
            self.client.transition_model_version_stage(
                name=self.model_name,
                version=current_prod['version'],
                stage="Archived"
            )
            print(f"📦 Versão {current_prod['version']} arquivada (era Production)")
        
        # Promover nova versão
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Production"
        )
        
        print(f"🚀 Versão {version} promovida para Production!")
        return True
    
    def archive_version(self, version: str) -> bool:
        """Move uma versão para Archived."""
        self.client.transition_model_version_stage(
            name=self.model_name,
            version=version,
            stage="Archived"
        )
        print(f"📦 Versão {version} arquivada")
        return True
    
    def delete_version(self, version: str) -> bool:
        """Deleta uma versão do registry."""
        self.client.delete_model_version(
            name=self.model_name,
            version=version
        )
        print(f"🗑️  Versão {version} deletada")
        return True
    
    def get_model_uri(self, stage: str = "Production") -> Optional[str]:
        """
        Retorna o URI do modelo em uma determinada stage.
        """
        versions = self.list_versions(stage=stage)
        if versions:
            version = versions[0]['version']
            return f"models:/{self.model_name}/{stage}"
        return None
    
    def print_status(self):
        """Imprime status atual do registry."""
        print(f"\n📋 Status do Registry: '{self.model_name}'")
        print("-" * 50)
        
        all_versions = self.list_versions()
        if not all_versions:
            print("   Nenhuma versão registrada")
            return
        
        for v in all_versions:
            stage_icon = {
                "Production": "🚀",
                "Staging": "🧪",
                "Archived": "📦",
                "None": "❓"
            }.get(v['stage'], "❓")
            
            print(f"   {stage_icon} v{v['version']} | {v['stage']} | run={v['run_id'][:8]}...")
        
        print("-" * 50)


def promote_pipeline(run_id: str, skip_staging: bool = False) -> str:
    """
    Pipeline completo de promoção.
    1. Registra run como nova versão → Staging
    2. (Opcional) Promove Staging → Production
    """
    manager = RegistryManager()
    
    print("=" * 60)
    print("🚀 MLflow Registry: Pipeline de Promoção")
    print("=" * 60)
    
    # Status antes
    manager.print_status()
    
    # Passo 1: Staging
    version = manager.promote_to_staging(run_id)
    
    # Passo 2: Production (se não skip)
    if not skip_staging:
        manager.promote_to_production(version)
    
    # Status depois
    manager.print_status()
    
    return version


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Uso: python registry_manager.py <run_id> [--skip-staging]")
        print("Exemplo: python registry_manager.py abc123")
        sys.exit(1)
    
    run_id = sys.argv[1]
    skip_staging = "--skip-staging" in sys.argv
    
    promote_pipeline(run_id, skip_staging)
