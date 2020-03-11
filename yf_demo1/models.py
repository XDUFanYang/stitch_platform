from django.db import models

# Create your models here.
from django.db import models


# Create your models here.

class imginfo(models.Model):
    img = models.FileField(upload_to='media')
