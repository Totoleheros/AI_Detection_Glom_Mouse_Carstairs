/**
 * Batch import ResNet50 classifications for ALL slides in the project.
 * Saves each slide's image data after import.
 * Run from: Automate → Script Editor
 */

import qupath.lib.io.GsonTools
import java.nio.file.Files
import java.nio.file.Paths
import com.google.gson.JsonParser

def geojsonDir = "/Users/antonino/Desktop/MLGlom/results/geojson"
def project    = getProject()

project.getImageList().each { entry ->
    def imageName   = entry.getImageName().replaceAll("\\.[^.]+\$", "")
    def geojsonPath = "${geojsonDir}/${imageName}_classified.geojson"

    if (!new File(geojsonPath).exists()) {
        println "⚠ No GeoJSON for ${imageName} — skipping"
        return
    }

    println "→ Processing: ${imageName}"
    def imageData = entry.readImageData()
    def hierarchy = imageData.getHierarchy()

    hierarchy.removeObjects(hierarchy.getFlattenedObjectList(null)
        .findAll { !it.isRootObject() }, true)

    def json        = new String(Files.readAllBytes(Paths.get(geojsonPath)))
    def featuresArr = JsonParser.parseString(json).getAsJsonObject().getAsJsonArray("features")
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

        def className = "Unclassified"
        if (props.has("classification")) {
            def cls = props.getAsJsonObject("classification")
            if (cls.has("name")) className = cls.get("name").getAsString()
        }

        def annotation = qupath.lib.objects.PathObjects.createAnnotationObject(
            objects[0].getROI(), getPathClass(className))

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
    entry.saveImageData(imageData)
    println "  ✓ ${newAnnotations.size()} glomeruli — ${imageName}"
}

println "\n✓ All slides processed"
