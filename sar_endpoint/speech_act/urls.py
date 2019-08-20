from django.urls import path

from . import views

urlpatterns = [
        path('swda/',views.classify_swda, name='swda'),
        path('vrm/',views.classify_vrm, name='vrm'),
        ]
