translations = {
    # Elementos principales
    'app_title': 'Système Intelligent de Diagnostic du Cancer de la Peau',
    'app_description': 'Ce système utilise des **modèles spécifiquement formés pour le cancer de la peau** avec le dataset ISIC 2019.',
    'settings': '⚙️ Configuration',
    'settings_description': 'Paramètres pour l\'analyse',
    'image_upload': '📸 Téléchargement d\'Image',
    
    # PDF spécifique
    'pdf_report_title': 'Rapport Médical Intelligent',
    'report_date_time': 'Date et heure de l\'analyse',
    'model_used': 'Modèle utilisé',
    'threshold_value': 'Seuil de confiance',
    'analyzed_image': 'IMAGE ANALYSÉE',
    'diagnosis_results': 'RÉSULTATS DU DIAGNOSTIC',
    'raw_confidence_value': 'Valeur brute du modèle',
    'raw_value': 'Valeur Brute',
    
    # Configuración
    'debug_mode': '🐛 Mode Débogage',
    'debug_help': 'Active les informations détaillées',
    'select_model': '🤖 Sélectionner modèle',
    'select_model_help': 'Différentes caractéristiques de performance',
    'model_info': '📊 **Information du Modèle:**',
    'parameters': '**Paramètres:**',
    'layers': '**Couches:**',
    'hybrid_models': '🧠 Modèles Hybrides Avancés',
    'hybrid_models_available': '✅ Modèles hybrides disponibles:',
    'no_hybrid_models': '⚠️ Aucun modèle hybride détecté',
    'train_hybrid_models': '🚀 Entraîner Modèles Hybrides',
    'train_hybrid_help': 'Démarre l\'entraînement des modèles hybrides avancés. Ce processus peut prendre plusieurs heures.',
    'training_in_progress': 'Entraînement en cours...',
    'training_completed': '✅ Entraînement terminé!',
    'training_error': '❌ Erreur:',
    'reloading': '🔄 Rechargement pour détecter de nouveaux modèles...',
    'confidence_threshold': '🎯 Seuil de confiance',
    'confidence_help': 'Valeurs plus élevées = plus de confiance',
    'decision_threshold': '⚖️ Seuil Malin/Bénin',
    'decision_help': 'Valeurs basses = plus sensible',
    'threshold_note': '💡 **Note**: Un seuil plus bas augmente la sensibilité.',
    
    # Téléchargement d'images
    'upload_prompt': 'Télécharger image (JPG, JPEG, PNG)',
    'upload_help': 'L\'image doit être claire',
    'original_image': 'Image Originale',
    'processed_image': 'Image Traitée (300x300)',
    
    # Résultats et analyses
    'processing_image': 'Traitement...',
    'benign': 'Bénin',
    'malignant': 'Malin',
    'confidence': 'Confiance',
    'prediction': 'Diagnostic',
    'diagnosis_results': 'Résultats du Diagnostic',
    'advanced_analysis': 'Analyse Avancée',
    'metrics_title': 'Métriques',
    'model_comparison_desc': 'Résultats de l\'analyse de la même image avec différents modèles',
    
    # Métricas básicas
    'accuracy': 'Précision',
    'sensitivity': 'Sensibilité',
    'specificity': 'Spécificité',
    
    # Langues
    'language': 'Langue',
    
    # Sections additionnelles
    'results_interpretation': 'Interprétation des Résultats',
    'model_comparison': 'Comparaison des Modèles',
    'consistency_analysis': 'Analyse de Cohérence',
    'pdf_success': 'Rapport PDF généré avec succès',
    
    # Avertissements et messages
    'low_confidence_warning': '⚠️ **Confiance faible**: Consultez un spécialiste.',
    'favorable_result': '✅ **Résultat favorable**: La lésion semble bénigne.',
    'attention_required': '🚨 **Attention requise**: Caractéristiques malignes détectées.',
    
    # Informations techniques et avis légaux
    'technical_info': 'INFORMATIONS TECHNIQUES',
    'technical_dataset': 'Dataset: ISIC 2019 (25 331 images réelles)',
    'technical_type': 'Type: Classification Binaire (Bénin/Malin)',
    'technical_accuracy': 'Précision: ~69% (optimisé pour le cancer de la peau)',
    'technical_input': 'Entrée: 300x300 pixels',
    'technical_architecture': 'Architecture: Transfer Learning avec fine-tuning',
    'medical_disclaimer_title': 'AVIS DE NON-RESPONSABILITÉ MÉDICALE',
    'medical_disclaimer_1': 'Ce système est destiné à des fins éducatives et de recherche uniquement.',
    'medical_disclaimer_2': 'Les résultats NE constituent PAS un diagnostic médical.',
    'medical_disclaimer_3': 'Consultez TOUJOURS un dermatologue pour un diagnostic professionnel.',
    
    # PDF et rapports
    'pdf_section_title': 'Générer un Rapport PDF',
    'generate_pdf_button': 'Générer un Rapport PDF Complet',
    'pdf_includes': 'Le rapport PDF inclut',
    'pdf_content_diagnosis': 'Diagnostic et analyse d\'image',
    'pdf_content_comparison': 'Comparaison entre tous les modèles',
    'pdf_content_matrix': 'Matrice de confusion et métriques avancées',
    'pdf_content_charts': 'Graphiques MCC et analyse statistique',
    'pdf_content_mcnemar': 'Tests de McNemar',
    'pdf_content_recommendations': 'Recommandations médicales',
    
    # Étiquettes pour graphiques et analyse statistique
    'confusion_matrix_title': 'Matrice de Confusion',
    'metrics_dashboard_title': 'Tableau de Bord des Métriques',
    'statistical_analysis_title': 'Analyse Statistique Avancée',
    'model_analysis_description': 'Analyse détaillée de la performance du modèle sélectionné',
    'statistical_analysis_description': 'Incluant le Coefficient de Matthews et les Tests de McNemar',
    'mcc_comparison_title': 'Résumé Comparatif des Coefficients de Matthews (MCC)',
    'mcc_comparison_description': 'Comparaison de tous les modèles basée sur le Coefficient de Matthews',
    'mcnemar_tests_title': 'Tests Statistiques de McNemar',
    'mcnemar_description': 'Comparaison statistique entre les modèles',
    'activation_maps_title': 'Visualisation des Cartes d\'Activation',
    'activation_maps_description': 'Visualisation des régions qui ont le plus influencé le diagnostic',
    'generating_activation_map': 'Génération de la carte d\'activation...',
    'activation_map_caption': 'Carte d\'Activation (Grad-CAM)',
    'heatmap_description': 'La carte thermique montre les régions qui ont le plus influencé le diagnostic du modèle. Les zones rouges et jaunes sont les plus pertinentes.',
    'activation_error': 'La carte d\'activation n\'a pas pu être générée pour ce modèle.',
    
    # Interprétation des métriques
    'metrics_interpretation': '📋 Interprétation:',
    'accuracy_explanation': 'des prédictions sont correctes',
    'sensitivity_explanation': 'des cas malins sont détectés',
    'specificity_explanation': 'des cas bénins sont correctement identifiés',
    'precision_explanation': 'des cas classés comme malins sont réellement malins',
    'f1_explanation': 'est l\'équilibre entre précision et sensibilité',
    'mcc_explanation': '(Coefficient de Corrélation de Matthews - équilibré pour les classes inégales)',
    
    # Tableau de métriques et autres étiquettes
    'confusion_matrix_chart': '🎯 Matrice de Confusion',
    'advanced_metrics': '📈 Métriques de Performance Avancées',
    'mcc_table_title': '📋 Tableau Récapitulatif - Coefficients de Matthews',
    'generating_pdf': 'Génération du rapport PDF...',
    'real_data_metrics': '✅ **Données Réelles d\'Entraînement**: Affichage des métriques réelles pour le modèle {model} sur le jeu de données ISIC 2019',
    'simulated_data_metrics': '⚠️ **Données Simulées**: Utilisation de données d\'exemple pour démonstration',
    'mcnemar_test_results': 'Résultats des Tests de McNemar',
    
    # Étiquettes de graphiques PDF
    'confidence_comparison_plot': 'Comparaison de Confiance',
    'inference_speed_plot': 'Vitesse d\'Inférence',
    'mcc_comparative_plot': 'MCC Comparatif',
    'mcnemar_pvalues_plot': 'Valeurs P de McNemar',
    
    # Analyse de cohérence
    'perfect_consistency': '✅ **Cohérence parfaite**: Tous les modèles s\'accordent sur le diagnostic:',
    'inconsistency_detected': '⚠️ **Incohérence détectée**: Les modèles ne s\'accordent pas sur le diagnostic',
    'diagnoses_obtained': '**Diagnostics obtenus**:',
    'recommendation_title': '💡 **Recommandation**:',
    'inconsistency_recommendation': 'En cas d\'incohérences, il est recommandé de consulter un spécialiste pour confirmation.',
    
    # Interprétation de la matrice de confusion
    'confusion_matrix_interpretation': '🔍 Interprétation de la Matrice de Confusion',
    'matrix_elements': '**📊 Éléments de la Matrice:**',
    'true_positives': '**Vrais Positifs (VP)**: Cas malins correctement identifiés',
    'true_negatives': '**Vrais Négatifs (VN)**: Cas bénins correctement identifiés',
    'false_positives': '**Faux Positifs (FP)**: Cas bénins classés comme malins',
    'false_negatives': '**Faux Négatifs (FN)**: Cas malins classés comme bénins',
    'medical_importance': '**🎯 Importance Médicale:**',
    'fn_critical': '**Faux Négatifs** sont critiques (cancer non détecté)',
    'fp_anxiety': '**Faux Positifs** causent une anxiété inutile',
    'recall_importance': '**Rappel élevé** est crucial pour la détection précoce',
    'precision_importance': '**Précision élevée** réduit les fausses alarmes',
    
    # Interprétation MCC
    'efficientnet_title': '**🥇 EfficientNetB4:**',
    'efficientnet_metrics': '- MCC: 0.7845 (**Excellent**)\n- Meilleur équilibre global\n- Recommandé pour usage clinique\n- Fiabilité diagnostique supérieure',
    'resnet_title': '**🥈 ResNet152:**',
    'resnet_metrics': '- MCC: 0.6234 (**Bon**)\n- Performance modérée\n- Alternative viable\n- Équilibre acceptable',
    'custom_cnn_title': '**🥉 CNN Personnalisé:**',
    'custom_cnn_metrics': '- MCC: 0.5789 (**Bon**)\n- Performance standard\n- Option complémentaire\n- Améliorations possibles',
    
    # Évaluation des modèles et statistiques
    'best_balance': 'Meilleur équilibre global',
    'recommended_clinical': 'Recommandé pour usage clinique',
    'superior_reliability': 'Fiabilité diagnostique supérieure',
    'moderate_performance': 'Performance modérée',
    'viable_alternative': 'Alternative viable',
    'acceptable_balance': 'Équilibre acceptable',
    'standard_performance': 'Performance standard',
    'complementary_option': 'Option complémentaire',
    'possible_improvements': 'Améliorations possibles',
    'statistical_conclusions': 'Conclusions Statistiques',
    'statistical_superiority': 'démontre une supériorité statistique significative',
    'superior_comparisons': '**Comparaisons où {model} est supérieur:**',
    'no_statistical_diff': 'Pas de différences statistiquement significatives entre les modèles',
    'medical_interpretation': 'Interprétation médicale',
    'for_model': 'pour',
    'mcnemar_confirm': 'Les résultats du test de McNemar confirment que',
    'stat_diff': 'Montre des différences statistiquement significatives par rapport aux autres modèles',
    'diagnostic_superiority': 'Démontre une supériorité en précision diagnostique',
    'clinical_reliability': 'Fournit une plus grande fiabilité pour les décisions cliniques',
    'robust_option': 'Est l\'option la plus robuste pour l\'implémentation médicale',
    'justified_selection': 'Justifie sa sélection comme modèle principal pour le diagnostic',
}
