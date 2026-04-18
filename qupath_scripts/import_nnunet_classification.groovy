/**
 * Import ResNet50 classification GeoJSON as annotations with measurements.
 * Each annotation gets: class (primary), name (all classes), probability scores.
 * Clears existing annotations first.
 * Change geojsonPath for each slide.
 */

import qupath.lib.io.GsonTools
import java.nio.file.Files
import java.nio.file.Paths
import com.google.gson.JsonParser

// ← Change for each slide
def geojsonPath = "/Users/antonino/Desktop/MLGlom/results/geojson/LysM_01_classified.geojson"

def hierarchy = getCurrentImageData().getHierarchy()
hierarchy.removeObjects(hierarchy.getFlattenedObjectList(null)
    .findAll { !it.isRootObject() }, true)

def json        = new String(Files.readAllBytes(Paths.get(geojsonPath)))
def root        = JsonParser.parseString(json).getAsJsonObject()
def featuresArr = root.getAsJsonArray("features")
def newAnnotations = []

featuresArr.each { featElem ->
    def feat  = featElem.getAsJsonObject()
    def props = feat.getAsJsonObject("properties")

    def singleFC = """{"type":"FeatureCollection","features":[${featElem.toString()}]}"""
    def objects  = GsonTools.getInstance().fromJson(
        JsonParser.parseString(singleFC).getAsJsonObject().getAsJsonArray("features"),
        new com.google.gson.reflect.TypeToken<List<qupath.lib.objects.PathObject>>(){}.getType()
    )
    if (!objects) return

    def roi = objects[0].getROI()

    def className = "Unclassified"
    if (props.has("classification")) {
        def cls = props.getAsJsonObject("classification")
        if (cls.has("name")) className = cls.get("name").getAsString()
    }

    def annotation = qupath.lib.objects.PathObjects.createAnnotationObject(
        roi, getPathClass(className))

    if (props.has("glom_classes"))
        annotation.setName(props.get("glom_classes").getAsString())

    def ml = annotation.getMeasurementList()
    ["prob_Normal","prob_Adhesion","prob_Thickening_GBM","prob_Fibrinoid_necrosis",
     "prob_Hypercellularity","prob_Fibrosis","prob_Crescent","prob_Sclerosis","n_classes"].each { key ->
        if (props.has(key)) ml.put(key, props.get(key).getAsDouble())
    }
    ml.close()
    newAnnotations << annotation
}

hierarchy.addObjects(newAnnotations)
fireHierarchyUpdate()
println "✓ ${newAnnotations.size()} classified glomeruli imported with probability measurements"
