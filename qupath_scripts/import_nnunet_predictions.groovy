/**
 * Import nnU-Net detection GeoJSON as Glomerulus annotations.
 * Clears existing annotations first.
 * Change geojsonPath for each slide.
 */

import qupath.lib.io.GsonTools
import java.nio.file.Files
import java.nio.file.Paths
import com.google.gson.JsonParser

// ← Change for each slide
def geojsonPath = "/Users/antonino/Desktop/GlomAndreMarc/detections_nnunet/LysM_01_detections.geojson"

def hierarchy = getCurrentImageData().getHierarchy()
hierarchy.removeObjects(hierarchy.getFlattenedObjectList(null)
    .findAll { !it.isRootObject() }, true)

def json        = new String(Files.readAllBytes(Paths.get(geojsonPath)))
def featuresArr = JsonParser.parseString(json).getAsJsonObject().getAsJsonArray("features")
def objects     = GsonTools.getInstance().fromJson(featuresArr,
    new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>(){}.getType())

hierarchy.addObjects(objects.collect {
    qupath.lib.objects.PathObjects.createAnnotationObject(
        it.getROI(), getPathClass("Glomerulus"))
})
fireHierarchyUpdate()
println "✓ ${objects.size()} annotations imported"
